use futures::{FutureExt, pin_mut, select};
use smol::Async;
use smol::channel::{Receiver, Sender, unbounded};
use smol::io::AsyncWriteExt;
use smol::net::unix::{UnixListener, UnixStream};
use std::ffi::{c_float, c_int, c_uint, c_void};
use std::hint::black_box;
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::SocketAddr;
use std::os::unix::net::UnixListener as StdUnixListener;
use std::ptr::null_mut;
use std::sync::OnceLock;
// use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

macro_rules! opaque_type {
    {$vis:vis $name:ident} => {
        #[repr(C)]
        #[allow(non_camel_case_types)]
        $vis struct $name {
            _data: (),
            _marker: ::core::marker::PhantomData<(*mut u8, ::core::marker::PhantomPinned)>,
        }
    };
}

// opaque_type! {pub CUfunc_st}

// type CUfunction = *mut CUfunc_st;

opaque_type! {pub CUstream_st}
type CUstream = *mut CUstream_st;

opaque_type! {pub CUevent_st}
type CUevent = *mut CUevent_st;

type CudaErrorT = c_int;

type CudaStreamCaptureStatus = c_int;

const CUDA_EVENT_BLOCKING_SYNC: c_uint = 0x1;

#[repr(C)]
#[derive(Copy, Clone)]
struct Dim3 {
    x: c_uint,
    y: c_uint,
    z: c_uint,
}

#[link(name = "cudart")]
unsafe extern "C" {
    //cudaError_t cudaEventCreate ( cudaEvent_t* event )
    fn cudaEventCreateWithFlags(event: *mut CUevent, flags: c_uint) -> CudaErrorT;
    fn cudaEventRecord(event: CUevent, stream: CUstream) -> CudaErrorT;
    // __host__​cudaError_t cudaEventSynchronize ( cudaEvent_t event )
    fn cudaEventSynchronize(event: CUevent) -> CudaErrorT;
    // ​cudaError_t cudaEventElapsedTime_v2 ( float* ms, cudaEvent_t start, cudaEvent_t end )
    fn cudaEventElapsedTime_v2(ms: *mut c_float, start: CUevent, end: CUevent) -> CudaErrorT;
    fn cudaEventElapsedTime(ms: *mut c_float, start: CUevent, end: CUevent) -> CudaErrorT;
    //cudaError_t cudaStreamIsCapturing ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus )
    fn cudaStreamIsCapturing(
        stream: CUstream,
        pCaptureStatus: *mut CudaStreamCaptureStatus,
    ) -> CudaErrorT;
}

struct KernelDescription {
    ev1: CUevent,
    ev2: CUevent,
    id: u32,
}

unsafe impl Send for KernelDescription {}

static CELL: OnceLock<Sender<KernelDescription>> = OnceLock::new();

// FIXME - we don't clean up the events after handling them here.
async fn serve_stream(mut stream: UnixStream, rx: &Receiver<KernelDescription>) {
    loop {
        let KernelDescription { ev1, ev2, id } = rx.recv().await.expect("XXX");
        unsafe {
            let err = cudaEventSynchronize(ev2);
            if err != 0 {
                panic!("{err}");
            }
            let mut ms: c_float = 0.0;
            let err = cudaEventElapsedTime(&mut ms, ev1, ev2);
            if err != 0 {
                panic!("{err}");
            }

            let mut buf = [0; 8];
            buf[0..4].copy_from_slice(&id.to_le_bytes());
            buf[4..8].copy_from_slice(&ms.to_le_bytes());

            if let Err(_) = stream.write_all(&buf).await {
                return;
            }
        }
    }
}

// FIXME - this is not going to work in containers,
// since it relies on the parca-agent being able to find things in
// the target process's network namespace.
async fn process_messages(rx: Receiver<KernelDescription>) {
    // The socket sends SIGPIPE when it's ignored, just ignore that.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };
    let name = format!("parcagpu.{}", std::process::id());
    let addr = SocketAddr::from_abstract_name(&name).unwrap();
    let l: UnixListener = Async::new(StdUnixListener::bind_addr(&addr).expect("XXX"))
        .expect("XXX")
        .into();
    loop {
        let stream_fut = l.accept().fuse();
        let rx_fut = rx.recv().fuse();
        pin_mut!(stream_fut, rx_fut);
        select! {
            // we got a connection, so handle it.
            res = stream_fut => {
                let (stream, _) = res.expect("XXX");
                // This blocks until the connection breaks
                serve_stream(stream, &rx).await;
            },
            // if we receive any messages while no parca-agent is connected,
            // just drop them here.
            res = rx_fut => {
                res.expect("XXX");
            }
        };
    }
}

type RealFn = unsafe extern "C" fn(
    *const c_void,
    Dim3,
    Dim3,
    *mut *mut c_void,
    usize,
    *mut CUstream_st,
) -> i32;

#[inline(never)]
#[unsafe(no_mangle)]
extern "C" fn shim_inner(
    id: u32,
    func: *const c_void,
    grid_dim: Dim3,
    block_dim: Dim3,
    args: *mut *mut c_void,
    shared_mem: usize,
    stream: CUstream,
    real: RealFn,
) -> CudaErrorT {
    black_box(id);
    unsafe { real(func, grid_dim, block_dim, args, shared_mem, stream) }
}

const SAMP: u32 = 1;

redhook::hook! {
    unsafe fn cudaLaunchKernel(
        func: *const c_void,
        grid_dim: Dim3,
        block_dim: Dim3,
        args: *mut *mut c_void,
        shared_mem: usize,
        stream: CUstream
    ) -> CudaErrorT => shim {
        let real = redhook::real!(cudaLaunchKernel);
        unsafe {
            let mut p_cap_stat = 0;
            let err = cudaStreamIsCapturing(stream, &raw mut p_cap_stat);
            if err != 0 {
                println!("Couldn't detect whether the stream is capturing: {err}");
                return err;
            }
            if p_cap_stat != 0 {
                println!("Stream {stream:p} is capturing; not recording the event.");
                return real(func, grid_dim, block_dim, args, shared_mem, stream);
            }
        }
        let mut rng = rand::rng();
        let id: u32 = rng.random();

        if id % SAMP != 0 {
            return unsafe { real(func, grid_dim, block_dim, args, shared_mem, stream) };
        }

        let kd = unsafe {
            let mut ev1: CUevent = std::ptr::null_mut();
            let err = cudaEventCreateWithFlags(&raw mut ev1, CUDA_EVENT_BLOCKING_SYNC);
            if err != 0 {
                return err;
            }
            let err = cudaEventRecord(ev1, stream);
            if err != 0 {
                return err;
            }
            let err = shim_inner(id, func, grid_dim, block_dim, args, shared_mem, stream, real);
            if err != 0 {
                return err;
            }
            let mut ev2: CUevent = std::ptr::null_mut();
            let err = cudaEventCreateWithFlags(&raw mut ev2, CUDA_EVENT_BLOCKING_SYNC);
            if err != 0 {
                return err;
            }
            let err = cudaEventRecord(ev2, stream);
            if err != 0 {
                return err;
            }
            KernelDescription {
                ev1, ev2, id
            }
        };

        let tx = CELL.get_or_init(|| {
            let (tx, rx) = unbounded();
            smol::spawn(process_messages(rx)).detach();

            tx
        });

        smol::block_on(tx.send(kd)).unwrap();

        0
    }
}

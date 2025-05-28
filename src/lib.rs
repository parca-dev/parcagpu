use std::ffi::{c_float, c_int, c_uint, c_void};
use std::ptr::null_mut;
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

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
    fn cudaEventCreate(event: *mut CUevent) -> CudaErrorT;
    fn cudaEventRecord(event: CUevent, stream: CUstream) -> CudaErrorT;
    // __host__​cudaError_t cudaEventSynchronize ( cudaEvent_t event )
    fn cudaEventSynchronize(event: CUevent) -> CudaErrorT;
    // ​cudaError_t cudaEventElapsedTime_v2 ( float* ms, cudaEvent_t start, cudaEvent_t end )
    fn cudaEventElapsedTime_v2(ms: *mut c_float, start: CUevent, end: CUevent) -> CudaErrorT;
    //cudaError_t cudaStreamIsCapturing ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus )
    fn cudaStreamIsCapturing(stream: CUstream, pCaptureStatus: *mut CudaStreamCaptureStatus) -> CudaErrorT;
}

// __host__​cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )

struct KernelDescription {
    ev1: CUevent,
    ev2: CUevent,
    id: u32,
}

unsafe impl Send for KernelDescription {}

static CELL: OnceLock<Sender<KernelDescription>> = OnceLock::new();

// XXX - cleanup
fn dostuff(rx: Receiver<KernelDescription>) {
    for KernelDescription { ev1, ev2, id } in rx {
        unsafe {
            let err = cudaEventSynchronize(ev2);
            if err != 0 {
                println!("wtf: {err}, {id:x}");
                continue;
            }
            let mut ms: c_float = 0.0;
            let err = cudaEventElapsedTime_v2(&raw mut ms, ev1, ev2);
            if err != 0 {
                println!("wtf: {err}, {id:x}");
            } else {
                println!("{id:x}: {ms}")
            }
        }
    }
}

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
        println!("Kernel launched: {func:p}; id: 0x{id:x}, stream {stream:p}");

        let kd = unsafe {
            let mut ev1: CUevent = std::ptr::null_mut();
            let err = cudaEventCreate(&raw mut ev1);
            if err != 0 {
                return err;
            }
            let err = cudaEventRecord(ev1, stream);
            if err != 0 {
                return err;
            }
            let err = real(func, grid_dim, block_dim, args, shared_mem, stream);
            if err != 0 {
                return err;
            }
            let mut ev2: CUevent = std::ptr::null_mut();
            let err = cudaEventCreate(&raw mut ev2);
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
            let (tx, rx) = channel();

            thread::spawn(move  || dostuff(rx));
            tx
        });

        tx.send(kd).unwrap();

        0
    }


    // CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra )
    // unsafe fn cuLaunchKernel(
    //     f: CUfunction,
    //     grid_dim_x: c_uint,
    //     grid_dim_y: c_uint,
    //     grid_dim_z: c_uint,
    //     shared_mem_bytes: c_uint,
    //     h_stream: CUstream,
    //     kernel_params: *mut *mut c_void,
    //     extra: *mut *mut c_void
    // ) -> CUresult => shim {
    //     println!("Kernel launched: {f:p}");
    //     unsafe {
    //         redhook::real!(cuLaunchKernel)(f, grid_dim_x, grid_dim_y, grid_dim_z, shared_mem_bytes, h_stream, kernel_params, extra)
    //     }
    // }

}

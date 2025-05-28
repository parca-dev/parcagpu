// CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra )

use std::ffi::c_void;

use libc::c_uint;

type CUresult = libc::c_int;

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

opaque_type! {pub CUfunc_st}

type CUfunction = *mut CUfunc_st;

opaque_type! {pub CUstream_st}

type CUstream = *mut CUstream_st;

redhook::hook! {
    unsafe fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        h_stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void
    ) -> CUresult => shim {
        println!("Kernel launched: {f:p}");
        unsafe {
            redhook::real!(cuLaunchKernel)(f, grid_dim_x, grid_dim_y, grid_dim_z, shared_mem_bytes, h_stream, kernel_params, extra)
        }
    }
}

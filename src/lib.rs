use std::ffi::{c_int, c_uint, c_void};

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

type CudaErrorT = c_int;

#[repr(C)]
#[derive(Copy, Clone)]
struct Dim3 {
    x: c_uint,
    y: c_uint,
    z: c_uint,
}

// __host__​cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )

redhook::hook! {
    unsafe fn cudaLaunchKernel(
        func: *const c_void,
        grid_dim: Dim3,
        block_dim: Dim3,
        args: *mut *mut c_void,
        shared_mem: usize,
        stream: CUstream
    ) -> CudaErrorT => shim {
        println!("Kernel launched: {func:p}");
        unsafe {
            redhook::real!(cudaLaunchKernel)(func, grid_dim, block_dim, args, shared_mem, stream)
        }
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

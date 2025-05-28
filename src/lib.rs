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

opaque_type! {CUfunc_st}

type CUfunction = *mut CUfunc_st;

opaque_type! {CUstream_st}

type CUstream = *mut CUstream_st;

unsafe extern "C" {
    fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}

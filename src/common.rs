extern crate libc;
extern crate std;
use self::libc::{c_int, c_double};

#[repr(C)]
#[derive(Copy, Debug)]
pub struct Struct_feature_node {
    pub index: c_int,
    pub value: c_double,
}

impl Clone for Struct_feature_node {
    fn clone(&self) -> Self { *self }
}

impl Default for Struct_feature_node {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
}

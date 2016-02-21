use std::marker::PhantomData;
use common;
use std::sync::Arc;



pub struct PtrVec<'a, T: 'a> {
    pub vec: Vec<*const T>,
    phantom: PhantomData<&'a Vec<T>>
}

// TODO figure out how to use lifetimes
pub fn vec_to_ptrs<'a, T>(examples: &'a Vec<Vec<T>>)
    -> PtrVec<'a, T> {

    // let all_x_vec: Vec<*mut Struct_feature_node> = Vec::new();
    let mut first_x_vec: Vec<*const T> = Vec::with_capacity(examples.len());

    for i in 0..examples.len() {
        first_x_vec.push((&examples[i][..]).as_ptr());
    }
    PtrVec { vec: first_x_vec, phantom: PhantomData }
}

pub fn make_sparse_matrix(xs: &Vec<Arc<Vec<f64>>>)
    -> (Vec<Vec<common::Struct_feature_node>>, i32) {

    let mut examples: Vec<Vec<common::Struct_feature_node>> =
        Vec::with_capacity(xs.len());
    
    let mut max_index = 1;
    for example in xs {
        let mut features: Vec<common::Struct_feature_node> = Vec::new();
        let mut idx = 1; // liblinear is 1-based indexing
        for f in example.iter() {
            if *f != 0.0 {
                if idx > max_index {
                    max_index = idx;
                }
                let f_node = common::Struct_feature_node{index: idx, value: *f};
                features.push(f_node);
            }
            idx += 1;
        }
        features.push(common::Struct_feature_node{index: -1, value: 0.0}); // -1 indicates end of feature vector
        examples.push(features);
    }
    (examples, max_index)
}

pub fn make_sparse_vector(x: &Vec<f64>) -> Vec<common::Struct_feature_node> {
    let mut sparse_x: Vec<common::Struct_feature_node> = Vec::with_capacity(x.len());
    for (i, f) in x.iter().enumerate() {
        if *f != 0.0_f64 {
            sparse_x.push(common::Struct_feature_node { index: i as i32 + 1, value: *f });
        }
    }
    sparse_x.push(common::Struct_feature_node { index: -1, value: 0.0_f64});
    sparse_x
}

// pub fn ptr_to_vec(











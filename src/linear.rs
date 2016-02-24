use std::slice;
use std::sync::Arc;

use linear_ffi;
pub use linear_ffi::{Struct_parameter, L2R_LR, L1R_LR};
use linalg;
use common;
use util;

// #[derive(Default, Debug)]
pub struct Problem {
    pub num_examples: i32,
    pub max_index: i32,
    pub labels: Vec<f64>,
    pub examples: Vec<Vec<common::Struct_feature_node>>,
    pub example_ptrs: Vec<*const common::Struct_feature_node>,
    pub bias: f64,
    pub raw: linear_ffi::Struct_problem,
}


impl Problem {

    pub fn from_training_data(xs: &Vec<Arc<Vec<f64>>>, ys: &Vec<f64>) -> Problem {
        let (examples, max_index) = util::make_sparse_matrix(xs);
        let example_ptrs = util::vec_to_ptrs(&examples).vec;
        let labels = ys.clone();
        let raw = linear_ffi::Struct_problem {
            l: ys.len() as i32,
            n: max_index,
            y: labels.as_ptr(),
            x: (&example_ptrs[..]).as_ptr(),
            bias: -1.0
        };
        Problem {
            num_examples: ys.len() as i32,
            max_index: max_index,
            labels: labels,
            examples: examples,
            example_ptrs: example_ptrs,
            bias: -1.0,
            raw: raw,
        }
    }
}


#[derive(Default, Debug)]
#[allow(non_snake_case)] // to better match liblinear names
pub struct Parameters {
    pub solver_type: u32,
    pub eps: f64,
    pub C: f64,
    pub nr_weight: i32,
    pub weight_label: Option<Vec<i32>>,
    pub weight: Option<Vec<f64>>,
    pub p: f64
}

impl Parameters {

    #[allow(dead_code)]
    fn from_raw(mut param: linear_ffi::Struct_parameter) -> Parameters {

        let mut safe_params: Parameters = Parameters::default();
        unsafe {
            safe_params.solver_type = param.solver_type;
            safe_params.eps = param.eps;
            safe_params.C = param.C;
            safe_params.nr_weight = param.nr_weight;
            // TODO weight_label, weight could be null
            if !param.weight_label.is_null() {
                safe_params.weight_label = Some(slice::from_raw_parts(param.weight_label, safe_params.nr_weight as usize).to_vec());
            } else {
                safe_params.weight_label = None;
            }
            if !param.weight.is_null() {
                safe_params.weight = Some(slice::from_raw_parts(param.weight, safe_params.nr_weight as usize).to_vec());
            } else {
                safe_params.weight = None;
            }
            safe_params.p = param.p;
            linear_ffi::destroy_param(&mut param as *mut linear_ffi::Struct_parameter);
        }
        safe_params
    }
}

#[derive(Default, Debug)]
pub struct LogisticRegressionModel {
    pub params: Parameters,
    pub nr_class: i32,
    pub nr_feature: i32,
    pub w: Vec<f64>,
    pub label: Option<Vec<i32>>,
    pub bias: f64
}

impl LogisticRegressionModel {
    #[allow(dead_code)]
    fn from_raw(model: *const linear_ffi::Struct_model) -> LogisticRegressionModel {
        let safe_model: LogisticRegressionModel = unsafe {
            let _model = LogisticRegressionModel {
                params: Parameters::from_raw((*model).param),
                nr_class: (*model).nr_class,
                nr_feature: (*model).nr_feature,
                // w: slice::from_raw_parts((*model).w, max_index as usize).to_vec(),
                w: slice::from_raw_parts((*model).w, (*model).nr_feature as usize).to_vec(),
                label: if !(*model).label.is_null() {
                    Some(slice::from_raw_parts((*model).label, (*model).nr_class as usize).to_vec())
                } else {
                    None
                },
                bias: (*model).bias,
            };
            linear_ffi::free_and_destroy_model(&model as *const *const linear_ffi::Struct_model);
            _model
        };
        safe_model
    }

    pub fn logistic_regression_predict(&self, x: &Vec<f64>) -> f64 {
        let dot = linalg::dot(&self.w, x);
        let (pos, neg) = self.get_labels();
        let pred = if dot > 0_f64 { pos } else { neg };
        pred as f64
    }

    pub fn get_labels(&self) -> (f64, f64) {
        let l = self.label.as_ref().unwrap();
        if l.len() == 1 {
            if l[0] == 1 {
                (1.0, 0.0)
            } else if l[0] == 0 {
                (0.0, 1.0)
            } else {
                panic!(format!("invalid label: {}", l[0]));
            }
        } else if l.len() == 2 {
            (l[0] as f64, l[1] as f64)
        } else {
            panic!(format!("strange number of labels: {}", l.len()));
        }
    }
}

pub fn train_logistic_regression(prob: Problem,
                                 params: linear_ffi:: Struct_parameter) -> LogisticRegressionModel {

    let model: LogisticRegressionModel = unsafe {
        // params.C = if find_c {
        //     let start_c = params.C;
        //     let nr_fold = 4;
        //     let max_c = 1024;
        //     let mut best_c = 0.0f64;
        //     let mut best_rate = 0.0f64;
        //     params.C
        // } else {
        //     params.C
        // };
        let _model = linear_ffi::train(
                &prob.raw as *const linear_ffi::Struct_problem,
                &params as *const linear_ffi::Struct_parameter);
        LogisticRegressionModel::from_raw(_model)
    };
    model
}


use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use itertools::Itertools;

use ::rand::random;
use macroquad::{
    color::GREEN,
    prelude::*,
    window::{clear_background, next_frame},
};
use ordered_float::OrderedFloat;

use crate::{
    data::Dataset,
    linalg::Matrix,
    neural::{
        ActivationFunction::{self, ReLU, Tanh},
        NeuralNet, NeuralNetBuilder,
    },
    visual::{WindowPos, draw_digit, paint_brush},
};

mod algos;
mod data;
mod linalg;
mod neural;
mod visual;

pub fn conf() -> Conf {
    Conf {
        window_title: "Neural net".to_string(),
        window_width: 700,
        window_height: 700,
        ..Default::default()
    }
}

const KEY_NUMBERS: [KeyCode; 10] = [
    KeyCode::Key0,
    KeyCode::Key1,
    KeyCode::Key2,
    KeyCode::Key3,
    KeyCode::Key4,
    KeyCode::Key5,
    KeyCode::Key6,
    KeyCode::Key7,
    KeyCode::Key8,
    KeyCode::Key9,
];

// #[macroquad::main(conf)]
fn main() {
    // let (mut train, test) = Dataset::get_mnist();
    let finally = Dataset::load_csv_test("data/kaggle-competition/mnist/processed/test.csv");
    // let mut net = NeuralNetBuilder::new(vec![784, 512, 512, 10])
    //     .set_activation_functions(vec![ReLU, ReLU])
    //     .set_batch_size(1)
    //     .set_momentum(0.0)
    //     .set_learning_rate(0.01)
    //     .build();
    let mut net = NeuralNet::from_file("output/mnist-98_6%-accuracy.network");

    let mut bw = BufWriter::new(File::create("mnist_output.csv").unwrap());
    writeln!(bw, "ImageId,Label").unwrap();
    for (id, input) in finally {
        writeln!(bw, "{id},{}", net.classify(input)).unwrap();
    }

    // println!("test accuracy: {:.2}%", net.accuracy(&test) * 100.0);
    // println!("train accuracy: {:.2}%", net.accuracy(&train) * 100.0);
    // net.train(&mut train, 1);
    // println!("test accuracy: {:.2}%", net.accuracy(&test) * 100.0);
    // println!("train accuracy: {:.2}%", net.accuracy(&train) * 100.0);

    // net.save_to_file("output/mnist.network");
}

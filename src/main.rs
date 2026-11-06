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
        NeuralNetBuilder,
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

#[macroquad::main(conf)]
async fn main() {
    let (mut train, test) =
        Dataset::load_csv_train("data/kaggle-competition/mnist/processed/train.csv").split();
    let mut net = NeuralNetBuilder::new(vec![784, 256, 256, 10])
        .set_activation_functions(vec![ReLU, ReLU])
        .set_batch_size(1)
        .set_momentum(0.0)
        .set_learning_rate(0.01)
        .build();

    dbg!(train.data.len(), test.data.len());

    println!("test accuracy: {:.2}%", net.accuracy(&test) * 100.0);
    println!("train accuracy: {:.2}%", net.accuracy(&train) * 100.0);
    net.train(&mut train, 10);
    println!("test accuracy: {:.2}%", net.accuracy(&test) * 100.0);
    println!("train accuracy: {:.2}%", net.accuracy(&train) * 100.0);

    let finally = Dataset::load_csv_test("data/kaggle-competition/mnist/processed/test.csv");
    let f = File::create("mnist_output.csv").unwrap();
    let mut bw = BufWriter::new(f);
    writeln!(bw, "ImageId,Label").unwrap();
    for (id, input) in finally {
        let class = net.forward(Matrix::new_row(input));
        writeln!(
            bw,
            "{id},{}",
            class
                .data
                .into_iter()
                .enumerate()
                .max_by_key(|(_, x)| OrderedFloat(*x))
                .unwrap()
                .0
        )
        .unwrap();
    }
}

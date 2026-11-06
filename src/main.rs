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
    let (mut train, test) = Dataset::load_csv_train("data/kaggle/train_processed.csv").split();
    let mut net = NeuralNetBuilder::new(vec![11, 16, 16, 2])
        .set_activation_functions(vec![Tanh, Tanh])
        .set_batch_size(1)
        .set_momentum(0.0)
        .set_learning_rate(0.01)
        .build();

    dbg!(net.accuracy(&test));
    dbg!(net.accuracy(&train));
    net.train(&mut train, 10_000);
    dbg!(net.accuracy(&test));
    dbg!(net.accuracy(&train));

    let finally = Dataset::load_csv_test("data/kaggle/test_processed.csv");
    let f = File::create("titanic_output.csv").unwrap();
    let mut bw = BufWriter::new(f);
    writeln!(bw, "PassengerId,Survived").unwrap();
    for (id, input) in finally {
        let class = net.forward(Matrix::new_row(input));
        writeln!(bw, "{id},{}", (class.data[1] > 0.5) as u8).unwrap();
    }
}

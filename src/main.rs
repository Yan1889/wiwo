use std::{
    f32::consts::PI,
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
    ui::{hash, root_ui},
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

#[macroquad::main(conf)]
async fn main() {
    let (train, _) = Dataset::get_mnist();

    let digit_original = train.data[0].0.clone();

    let net = NeuralNet::from_file("output/mnist-augmented.network");
    let win = WindowPos::full();

    let mut angle = 0.0;
    let mut scale = 1.0;
    let mut transform = (0.0, 0.0);

    loop {
        let mut digit_modified = digit_original.clone();
        Dataset::transform_digit(
            &mut digit_modified,
            angle as f64,
            scale as f64,
            (transform.0 as f64, transform.1 as f64),
        );

        root_ui().slider(hash!(), "angle", -PI / 1.0..PI / 1.0, &mut angle);
        root_ui().slider(hash!(), "scale", 0.3..2.0, &mut scale);
        root_ui().slider(hash!(), "transform x", -9.0..9.0, &mut transform.0);
        root_ui().slider(hash!(), "transform y", -9.0..9.0, &mut transform.1);

        draw_digit(&digit_modified, win);

        if is_key_pressed(KeyCode::Enter) {
            println!("predicted: {}", net.classify(digit_modified))
        }
        next_frame().await
    }
}

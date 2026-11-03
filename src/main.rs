use std::{
    fs::File,
    io::Write,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use itertools::Itertools;
use macroquad::prelude::*;

use macroquad::{
    color::GREEN,
    prelude::*,
    window::{clear_background, next_frame},
};

use crate::{
    data::Dataset,
    neural::{
        ActivationFunction::{self, ReLU},
        NeuralNetBuilder,
    },
    visual::{WindowPos, draw_digit},
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

#[macroquad::main(conf)]
async fn main() {
    let (train, test) = Dataset::get_mnist();
    let win = WindowPos::full();
    loop {
        draw_digit(&test.data[0].0, win);
        next_frame().await
    }
}

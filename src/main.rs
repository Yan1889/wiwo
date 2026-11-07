use std::{
    f32::consts::PI,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use itertools::Itertools;

use macroquad::{
    color::GREEN,
    input::{KeyCode, is_key_pressed},
    text::draw_text,
    ui::{hash, root_ui},
    window::{Conf, clear_background, next_frame},
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
#[macroquad::main(conf)]
async fn main() {
    // inspect_human_digits().await;
    collect_human_digits().await;
}

async fn inspect_human_digits() {
    let human = Dataset::load_csv_train("output/human-test.csv");
    let net = NeuralNet::from_file("output/mnist-augmented.network");

    println!("accuracy on human: {:.2}%", net.accuracy(&human));

    let win = WindowPos::full();
    let mut i = 0;
    loop {
        let actual = human.data[i].1;
        let prediction = net.classify(human.data[i].0.clone());

        draw_digit(&human.data[i].0, win);
        draw_text(format!("actual: {actual}"), 50.0, 50.0, 30.0, GREEN);
        draw_text(
            format!("predicted ({}): {prediction}", prediction == actual),
            50.0,
            100.0,
            30.0,
            GREEN,
        );

        if is_key_pressed(KeyCode::Enter) {
            i += 1;
        }
        next_frame().await;
    }
}

async fn collect_human_digits() {
    let mut data = vec![0.0; 784];
    let win = WindowPos::full();
    let mut label = 0;

    let mut bw = BufWriter::new(
        OpenOptions::new()
            .append(true)
            .open("output/human-test.csv")
            .unwrap(),
    );

    loop {
        paint_brush(&mut data, win);
        draw_text(label.to_string(), 30.0, 30.0, 50.0, GREEN);
        if is_key_pressed(KeyCode::Enter) {
            writeln!(bw, "{label},{}", data.iter().join(",")).unwrap();
            bw.flush().unwrap();
            label = (label + 1) % 10;
            data.fill(0.0);
        }
        next_frame().await
    }
}

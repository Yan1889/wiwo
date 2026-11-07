use std::{
    f32::consts::PI,
    fs::{File, FileType, OpenOptions},
    io::{BufWriter, Write},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use itertools::Itertools;

use macroquad::{
    color::{BLUE, GREEN},
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
    visual::{WindowPos, draw_digit, draw_prediction, paint_brush},
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
// #[macroquad::main(conf)]
fn main() {
    // inspect_human_digits().await;
    // collect_human_digits().await;
    // play().await;
    // sliders().await;
    train_model();
}

fn train_model() {
    let (mut train, mut test) = Dataset::get_mnist();
    train.randomize_mnist();
    test.randomize_mnist();

    let mut net = NeuralNetBuilder::new(vec![784, 512, 512, 10])
        .set_activation_functions(vec![ReLU, ReLU])
        .set_momentum(0.0)
        .set_learning_rate(0.001)
        .set_batch_size(64)
        .build();

    println!("test accuracy: {:.2}%", net.accuracy(&test) * 100.0);
    println!("train accuracy: {:.2}%", net.accuracy(&train) * 100.0);
    net.train(&mut train, 20);
    println!("test accuracy: {:.2}%", net.accuracy(&test) * 100.0);
    println!("train accuracy: {:.2}%", net.accuracy(&train) * 100.0);

    net.save_to_file("output/neural.network");
}

async fn sliders() {
    let (image, _) = Dataset::get_mnist().1.data[0].clone();
    let win = WindowPos::full();
    let mut angle = 0.0;
    let mut scale = (1.0, 1.0);
    let mut translate = (0.0, 0.0);
    loop {
        let mut image = image.clone();
        Dataset::transform_digit(
            &mut image,
            angle as f64,
            (scale.0 as f64, scale.1 as f64),
            (translate.0 as f64, translate.1 as f64),
        );
        draw_digit(&image, false, win);
        root_ui().slider(
            hash!(),
            "angle",
            -15.0 * PI / 180.0..15.0 * PI / 180.0,
            &mut angle,
        );
        root_ui().slider(hash!(), "scale x", 0.85..1.2, &mut scale.0);
        root_ui().slider(hash!(), "scale y", 0.85..1.2, &mut scale.1);
        root_ui().slider(hash!(), "translate x", -3.0..3.0, &mut translate.0);
        root_ui().slider(hash!(), "translate y", -3.0..3.0, &mut translate.1);
        next_frame().await
    }
}

async fn play() {
    let net = NeuralNet::from_file("output/mnist-augmented.network");
    let mut data = vec![0.0; 784];
    let [win_digit, win_prediction, win_heatmap, _] = WindowPos::split_quad();

    loop {
        paint_brush(&mut data, win_digit);
        draw_prediction(&data, &net, win_prediction);

        let mut sum = net.input_gradients(data.clone());
        for i in 0..784 {
            sum[i] *= data[i];
        }
        draw_digit(&sum, true, win_heatmap);
        next_frame().await;
    }
}

async fn inspect_human_digits() {
    let human = Dataset::load_csv_train("output/human-test.csv");

    let net = NeuralNet::from_file("output/mnist-augmented.network");

    // println!("accuracy on human: {:.2}%", net.accuracy(&human));

    for filename in std::fs::read_dir("output").unwrap() {
        let s = filename.unwrap();
        if s.file_type().unwrap().is_file() && s.path().extension().unwrap() == "network" {
            let net = NeuralNet::from_file(&s.path().into_os_string().into_string().unwrap());
            println!(
                "{}: {:.2}%",
                s.file_name().into_string().unwrap(),
                net.accuracy(&human)
            );
        }
    }

    let win = WindowPos::full();
    let mut i = 0;
    loop {
        let actual = human.data[i].1;
        let prediction = net.classify(human.data[i].0.clone());

        draw_digit(&human.data[i].0, false, win);
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
        draw_text(format!("goal: {label}"), 30.0, 50.0, 50.0, GREEN);

        if is_key_pressed(KeyCode::Enter) {
            writeln!(bw, "{label},{}", data.iter().join(",")).unwrap();
            bw.flush().unwrap();
            label = (label + 1) % 10;
            data.fill(0.0);
        }
        next_frame().await
    }
}

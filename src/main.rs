use std::f64::consts::TAU;

use macroquad::{
    color::{BLACK, BLUE, Color, RED, WHITE},
    shapes::{draw_circle, draw_line, draw_rectangle},
    text::draw_text,
    time::draw_fps,
    window::{clear_background, next_frame, screen_height, screen_width},
};

use crate::{
    data::Dataset,
    linalg::Matrix,
    neural::{ActivationFunction, NeuralNetBuilder},
};

mod data;
mod linalg;
mod neural;

#[macroquad::main("XOR-Loss")]
async fn main() {
    class_plane().await;
}

fn iris() {
    let (train, test) = Dataset::get_iris().split();

    let mut net = NeuralNetBuilder::new(vec![4, 10, 10, 3])
        .set_activation_functions(vec![
            ActivationFunction::Sigmoid,
            ActivationFunction::Sigmoid,
            ActivationFunction::Sigmoid,
        ])
        .set_learning_rate(0.03)
        .build();

    dbg!(net.mse(&test));
    net.train(&train, 50);
    dbg!(net.mse(&test));
}

async fn class_plane() {
    let mut net = NeuralNetBuilder::new(vec![2, 8, 8, 1])
        .set_activation_functions(vec![
            ActivationFunction::Tanh,
            ActivationFunction::Tanh,
            ActivationFunction::Sigmoid,
        ])
        .set_learning_rate(0.05)
        .build();

    let dataset = Dataset::get_heart();

    loop {
        net.train(&dataset, 10);

        clear_background(BLACK);

        let resolution = 300;
        for i in 0..=resolution {
            for j in 0..=resolution {
                let x = i as f32 / resolution as f32;
                let y = j as f32 / resolution as f32;
                let w = 1.0 / resolution as f32;
                let h = 1.0 / resolution as f32;
                let r = net.forward(Matrix::new_column(vec![x as f64, y as f64])).0[0][0];
                draw_rectangle(
                    x * screen_width(),
                    (1.0 - y) * screen_height(),
                    w * screen_width(),
                    h * screen_height(),
                    if (r - 0.5).abs() < 0.07 {
                        WHITE
                        // Color::new(1.0 - r as f32, 0.0, r as f32, 0.5)
                    } else {
                        Color::new(1.0 - r as f32, 0.0, r as f32, 0.5)
                    },
                );
            }
        }
        for (input, output) in dataset.data() {
            let x = input[0];
            let y = input[1];
            let l = output[0];
            draw_circle(
                x as f32 * screen_width(),
                (1.0 - y as f32) * screen_height(),
                0.003 * screen_width(),
                if l > 0.5 { BLUE } else { RED },
            );
        }
        draw_fps();
        next_frame().await
    }
}

async fn xor_plane() {
    let mut net = NeuralNetBuilder::new(vec![2, 3, 1])
        .set_activation_functions(vec![
            ActivationFunction::Sigmoid,
            ActivationFunction::Sigmoid,
        ])
        .set_learning_rate(0.1)
        .build();

    let data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    let dataset = Dataset::new(data.clone());
    loop {
        net.train(&dataset, 300);

        clear_background(RED);

        let resolution = 150;
        for i in 0..resolution {
            for j in 0..resolution {
                let x = i as f32 / resolution as f32;
                let y = j as f32 / resolution as f32;
                let w = 1.0 / resolution as f32;
                let h = 1.0 / resolution as f32;
                let r = net.forward(Matrix::new_column(vec![x as f64, y as f64])).0[0][0];
                draw_rectangle(
                    x * screen_width(),
                    (1.0 - y) * screen_height(),
                    w * screen_width(),
                    h * screen_height(),
                    if (r - 0.5).abs() < 0.02 {
                        WHITE
                    } else {
                        Color::new(r as f32, 1.0 - r as f32, 0.0, 1.0)
                    },
                );
            }
        }
        for (input, output) in dataset.data() {
            let x = input[0];
            let y = input[1];
            let l = output[0];
            draw_circle(
                x as f32 * screen_width(),
                (1.0 - y as f32) * screen_height(),
                0.03 * screen_width(),
                if l > 0.5 { BLUE } else { RED },
            );
        }
        draw_fps();
        next_frame().await
    }
}

async fn xor() {
    let mut net = NeuralNetBuilder::new(vec![2, 3, 1])
        .set_activation_functions(vec![
            ActivationFunction::Sigmoid,
            ActivationFunction::Sigmoid,
        ])
        .set_learning_rate(0.1)
        .build();

    let data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    let dataset = Dataset::new(data.clone());
    let mut history = vec![net.mse(&dataset) as f32];
    loop {
        net.train(&dataset, 1000);
        history.push(net.mse(&dataset) as f32);

        clear_background(RED);
        // normalize to [0; 1]
        let s = history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let l = history.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let transformed = history
            .iter()
            .map(|&x| (x - s) / (l - s))
            .collect::<Vec<f32>>();
        for (i, [a, b]) in transformed.array_windows().enumerate() {
            let x1 = i as f32 / history.len() as f32 * screen_width();
            let x2 = (i + 1) as f32 / history.len() as f32 * screen_width();
            let y1 = (1.0 - a) * screen_height();
            let y2 = (1.0 - b) * screen_height();
            draw_line(x1, y1, x2, y2, 3.0, BLACK);
        }
        draw_text(
            format!("{}", history.last().unwrap()),
            300.0,
            300.0,
            20.0,
            BLACK,
        );

        draw_fps();
        next_frame().await
    }
}

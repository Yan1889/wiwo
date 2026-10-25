use macroquad::{
    color::{BLACK, BLUE, Color, RED, WHITE},
    shapes::{draw_circle, draw_rectangle},
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

#[macroquad::main("Machine Learning")]
async fn main() {
    class_plane().await;
}

fn iris() {
    let (mut train, test) = Dataset::get_iris().split();

    let mut net = NeuralNetBuilder::new(vec![4, 10, 10, 3])
        .set_activation_functions(vec![
            ActivationFunction::Sigmoid,
            ActivationFunction::Sigmoid,
            ActivationFunction::Sigmoid,
        ])
        .set_learning_rate(0.03)
        .build();

    dbg!(net.mse(&test));
    net.train(&mut train, 50);
    dbg!(net.mse(&test));
}

async fn class_plane() {
    let mut net = NeuralNetBuilder::new(vec![2, 8, 8, 1])
        .set_activation_functions(vec![
            ActivationFunction::Tanh,
            ActivationFunction::Tanh,
            ActivationFunction::Sigmoid,
        ])
        .set_learning_rate(0.01)
        .set_batch_size(1)
        .build();

    let mut dataset = Dataset::get_spiral();

    let discrete = true;

    loop {
        net.train(&mut dataset, 10);

        clear_background(BLACK);
        let resolution = 100;
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
                    if discrete {
                        if r < 0.5 {
                            RED.with_alpha(0.5)
                        } else {
                            BLUE.with_alpha(0.5)
                        }
                    } else {
                        // RED -> White -> Blue
                        // 1, 0, 0 -> 1, 1, 1 -> 0, 0, 1
                        if r < 0.5 {
                            Color::new(1.0, 2.0 * r as f32, 2.0 * r as f32, 0.5)
                        } else {
                            Color::new(
                                1.0 - 2.0 * (r - 0.5) as f32,
                                1.0 - 2.0 * (r - 0.5) as f32,
                                1.0,
                                0.5,
                            )
                        }
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

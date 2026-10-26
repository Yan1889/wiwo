use macroquad::{
    color::{GREEN, WHITE},
    prelude,
    window::{clear_background, next_frame},
};

use crate::{
    data::Dataset,
    neural::{ActivationFunction, NeuralNetBuilder},
    visual::{WindowPos, draw_colored_plane, draw_loss},
};

mod data;
mod linalg;
mod neural;
mod visual;

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

    let (mut train, test) = Dataset::get_spiral().split();

    let mut losses = vec![net.mse(&test)];
    let [win_plane, win_loss] = WindowPos::split_v();
    loop {
        net.train(&mut train, 10);
        losses.push(net.mse(&test));

        clear_background(GREEN);
        draw_loss(&losses, win_loss);
        draw_colored_plane(&net, &train, false, win_plane);
        next_frame().await;
    }
}

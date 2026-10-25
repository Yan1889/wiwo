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

    let mut dataset = Dataset::get_cross();
    let mut losses = vec![net.mse(&dataset)];
    let [win1, win2] = WindowPos::split_v();
    loop {
        net.train(&mut dataset, 10);
        losses.push(net.mse(&dataset));
        clear_background(GREEN);
        draw_colored_plane(&net, &dataset, false, win1);
        draw_loss(&losses, win2);
        next_frame().await;
    }
}

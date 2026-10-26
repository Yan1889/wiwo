use macroquad::{
    color::GREEN,
    window::{clear_background, next_frame},
};

use crate::{
    data::Dataset,
    neural::{ActivationFunction, NeuralNetBuilder},
    visual::{WindowPos, draw_accuracy, draw_colored_plane, draw_loss},
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

    dbg!(net.cross_entropy(&test));
    net.train(&mut train, 50);
    dbg!(net.cross_entropy(&test));
}

async fn class_plane() {
    let mut net = NeuralNetBuilder::new(vec![4, 8, 8, 3])
        .set_activation_functions(vec![ActivationFunction::Tanh, ActivationFunction::Tanh])
        .set_learning_rate(0.01)
        .set_batch_size(1)
        .build();

    let (mut train, test) = Dataset::get_iris().split();

    let mut losses = vec![net.cross_entropy(&test)];
    let mut accuracy_test = vec![net.accuracy(&test)];
    let mut accuracy_train = vec![net.accuracy(&train)];

    let [win_loss, win_accuracy] = WindowPos::split_v();
    loop {
        net.train(&mut train, 10);
        losses.push(net.cross_entropy(&test));
        accuracy_test.push(net.accuracy(&test));
        accuracy_train.push(net.accuracy(&train));

        clear_background(GREEN);
        draw_loss(&losses, win_loss);
        draw_accuracy(&accuracy_train, &accuracy_test, win_accuracy);
        next_frame().await;
    }
}

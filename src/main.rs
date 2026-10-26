use macroquad::{
    color::GREEN,
    window::{clear_background, next_frame},
};

use crate::{
    data::Dataset,
    neural::{ActivationFunction, NeuralNetBuilder},
    visual::{Border, WindowPos, draw_accuracy, draw_colored_plane, draw_loss},
};

mod data;
mod linalg;
mod neural;
mod visual;

#[macroquad::main("Machine Learning")]
async fn main() {
    let mut net = NeuralNetBuilder::new(vec![2, 8, 8, 3])
        .set_activation_functions(vec![ActivationFunction::Tanh, ActivationFunction::Tanh])
        .set_learning_rate(0.01)
        .set_batch_size(10)
        .build();

    let (mut train, test) = Dataset::get_triple().split();

    let mut loss_test = vec![net.cross_entropy(&test)];
    let mut loss_train = vec![net.cross_entropy(&train)];
    let mut accuracy_test = vec![net.accuracy(&test)];
    let mut accuracy_train = vec![net.accuracy(&train)];

    let [win_loss, win_accuracy, win_plane_test, win_plane_train] = WindowPos::split_quad();
    // let [win_loss, win_accuracy] = WindowPos::split_v();
    loop {
        net.train(&mut train, 1);
        loss_test.push(net.cross_entropy(&test));
        loss_train.push(net.cross_entropy(&train));
        accuracy_test.push(net.accuracy(&test));
        accuracy_train.push(net.accuracy(&train));

        clear_background(GREEN);
        draw_loss(&loss_train, &loss_test, win_loss);
        draw_accuracy(&accuracy_train, &accuracy_test, win_accuracy);
        draw_colored_plane(&net, &train, Border::Lerp, win_plane_train);
        draw_colored_plane(&net, &test, Border::Discrete, win_plane_test);
        next_frame().await;
    }
}

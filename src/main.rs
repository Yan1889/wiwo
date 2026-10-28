use std::time::Instant;

use macroquad::{
    color::GREEN,
    window::{clear_background, next_frame},
};

use crate::{
    algos::knn_accuracy,
    data::Dataset,
    neural::{ActivationFunction, NeuralNetBuilder},
    visual::{
        Border, WindowPos, draw_accuracy, draw_colored_plane_knn, draw_colored_plane_net, draw_loss,
    },
};

mod algos;
mod data;
mod linalg;
mod neural;
mod visual;

#[macroquad::main("Machine Learning")]
async fn main() {
    /*
    let mut net = NeuralNetBuilder::new(vec![2, 16, 16, 2])
        .set_activation_functions(vec![ActivationFunction::Tanh, ActivationFunction::Tanh])
        .set_learning_rate(0.003)
        .set_batch_size(64)
        .set_momentum(0.9)
        .build();

    let (mut train, mut test) = Dataset::get_xor().split();

    let mut loss_test = vec![net.cross_entropy(&test)];
    let mut loss_train = vec![net.cross_entropy(&train)];
    let mut accuracy_test = vec![net.accuracy(&test)];
    let mut accuracy_train = vec![net.accuracy(&train)];
     */

    // let [win_loss, win_accuracy, win_plane_test, win_plane_train] = WindowPos::split_quad();
    // let [win_loss, win_accuracy] = WindowPos::split_v();
    // let [win1, win2, win3, win4] = WindowPos::split_quad();
    // let win = WindowPos::full();
    let (train, test) = Dataset::get_mnist();
    for k in [1, 3, 5, 7, 15] {
        let start = Instant::now();
        println!(
            "k = {k}: {:.2}%, took {:?}",
            knn_accuracy(&train, &test, k) * 100.0,
            start.elapsed()
        );
    }

    loop {
        // net.train(&mut train, 10);
        // loss_test.push(net.cross_entropy(&test));
        // loss_train.push(net.cross_entropy(&train));
        // accuracy_test.push(net.accuracy(&test));
        // accuracy_train.push(net.accuracy(&train));

        clear_background(GREEN);
        // draw_loss(&loss_train, &loss_test, win_loss);
        // draw_accuracy(&accuracy_train, &accuracy_test, win_accuracy);
        // draw_colored_plane_knn(&mut train, 1, win_plane_train);
        // draw_colored_plane_knn(&mut test, 1, win_plane_test);
        next_frame().await;
    }
}

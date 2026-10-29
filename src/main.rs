use std::{fs::File, io::Write, time::Instant};

use itertools::Itertools;
/*
use macroquad::{
    color::GREEN,
    window::{clear_background, next_frame},
};
*/

use crate::{
    algos::knn_accuracy,
    data::Dataset,
    neural::{ActivationFunction, NeuralNetBuilder},
    // visual::{
    //     Border, WindowPos, draw_accuracy, draw_colored_plane_knn, draw_colored_plane_net, draw_loss,
    // },
};

mod algos;
mod data;
mod linalg;
mod neural;
mod visual;

// #[macroquad::main("Machine Learning")]
fn main() {
    collect_data();
    return;
    /*
    let deep = vec![784, 256, 128, 64, 10];
    let shallow = vec![784, 16, 16, 10];
    let mut net = NeuralNetBuilder::new(shallow)
        .set_activation_functions(vec![ActivationFunction::ReLU, ActivationFunction::ReLU])
        .set_learning_rate(0.001)
        .set_batch_size(1)
        .set_momentum(0.0)
        .build();

    let (mut train, test) = Dataset::get_mnist();

    let mut loss_test = vec![net.cross_entropy(&test)];
    let mut loss_train = vec![net.cross_entropy(&train)];
    let mut accuracy_test = vec![net.accuracy(&test)];
    let mut accuracy_train = vec![net.accuracy(&train)];
    */

    // let [win_loss, win_accuracy, win_plane_test, win_plane_train] = WindowPos::split_quad();
    // let [win_loss, win_accuracy] = WindowPos::split_v();
    // let [win1, win2, win3, win4] = WindowPos::split_quad();
    // let win = WindowPos::full();
    /*
    let (train, test) = Dataset::get_mnist();
    for k in [1, 3, 5, 7, 15] {
        let start = Instant::now();
        println!(
            "k = {k}: {:.2}%, took {:?}",
            knn_accuracy(&train, &test, k, false) * 100.0,
            start.elapsed()
        );
    }
    */
    /*
    loop {
        net.train(&mut train, 1);
        loss_test.push(net.cross_entropy(&test));
        loss_train.push(net.cross_entropy(&train));
        accuracy_test.push(net.accuracy(&test));
        accuracy_train.push(net.accuracy(&train));

        clear_background(GREEN);
        draw_loss(&loss_train, &loss_test, win_loss);
        draw_accuracy(&accuracy_train, &accuracy_test, win_accuracy);
        // draw_colored_plane_knn(&mut train, 1, win_plane_train);
        // draw_colored_plane_knn(&mut test, 1, win_plane_test);
        next_frame().await;
    }
    */
}

pub fn collect_data() {
    let (mut train, test) = Dataset::get_mnist();

    let mut file = File::create(format!("output/results_{}.csv", rand::random::<u32>())).unwrap();
    file.write_all(b"activation_function,architecture,accuracy,time\n")
        .unwrap();

    // 2 activation functions: ReLU, Tanh
    // 3 architectures: 256-128-64 (large), 64-16 (medium), <nothing> (small)
    for activation_function in [ActivationFunction::ReLU, ActivationFunction::Tanh] {
        for mut architecture in [vec![], vec![64, 16], vec![256, 128, 64]] {
            // add input and output layers
            architecture.insert(0, 784);
            architecture.push(10);

            let mut net = NeuralNetBuilder::new(architecture.clone())
                .set_activation_functions(vec![activation_function; architecture.len() - 2])
                .build();
            let start = Instant::now();
            net.train(&mut train, 10);
            file.write_all(
                format!(
                    "{:?},{},{:.2}%,{:?}\n",
                    activation_function,
                    architecture.iter().join("-"),
                    net.accuracy(&test) * 100.0,
                    start.elapsed(),
                )
                .as_bytes(),
            )
            .unwrap();
        }
    }
}

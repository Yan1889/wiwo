use crate::{
    linalg::Matrix,
    neural::{ActivationFunction, NeuralNetBuilder},
};

mod linalg;
mod neural;

fn main() {
    for hidden in 2..=4 {
        let mut net = NeuralNetBuilder::new(vec![2, hidden, 1])
            .set_activation_functions(vec![ActivationFunction::Tanh, ActivationFunction::Sigmoid])
            .set_learning_rate(0.1)
            .build();

        let data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        for (input, _) in &data {
            // dbg!(net.forward(Matrix::new_column(input.clone())));
        }
        for _ in 0..10_000 {
            for (input, output) in &data {
                net.train(
                    Matrix::new_column(input.clone()),
                    Matrix::new_column(output.clone()),
                );
            }
        }
        println!("--------------");
        let mut loss = 0.0;
        for (input, output) in &data {
            let d = net.forward(Matrix::new_column(input.clone())).0[0][0] - output[0];
            loss += d * d;
            // dbg!(net.forward(Matrix::new_column(input.clone())));
        }
        println!("{hidden} hidden, loss = {loss}");
    }
}

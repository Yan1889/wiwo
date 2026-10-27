use std::time::Instant;

use crate::{data::Dataset, linalg::Matrix};

#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    LReLU(f64),
    Sigmoid,
    Tanh,
}

impl ActivationFunction {
    fn sigmoid(x: f64) -> f64 {
        1.0 / ((-x).exp() + 1.0)
    }
    fn map(self, mat: &Matrix) -> Matrix {
        mat.map(|x| self.apply(x))
    }
    fn map_derivative(self, mat: &Matrix) -> Matrix {
        mat.map(|x| self.derivative(x))
    }
    fn apply(self, x: f64) -> f64 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::LReLU(a) => {
                if x < 0.0 {
                    a * x
                } else {
                    x
                }
            }
            Self::Sigmoid => Self::sigmoid(x),
            Self::Tanh => x.tanh(),
        }
    }
    fn derivative(self, x: f64) -> f64 {
        match self {
            Self::ReLU => (x > 0.0) as u8 as f64,
            Self::LReLU(a) => {
                if x < 0.0 {
                    a
                } else {
                    1.0
                }
            }
            Self::Sigmoid => Self::sigmoid(x) * (1.0 - Self::sigmoid(x)),
            Self::Tanh => 1.0 - x.tanh() * x.tanh(),
        }
    }
}

pub struct NeuralNetBuilder {
    afs: Vec<ActivationFunction>,
    learning_rate: f64,
    layer_sizes: Vec<usize>,
    batch_size: usize,
}
impl NeuralNetBuilder {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        Self {
            afs: vec![ActivationFunction::Tanh; layer_sizes.len() - 2],
            learning_rate: 0.01,
            layer_sizes,
            batch_size: 1,
        }
    }
    pub fn set_activation_functions(mut self, afs: Vec<ActivationFunction>) -> Self {
        self.afs = afs;
        self
    }
    pub fn set_learning_rate(mut self, r: f64) -> Self {
        self.learning_rate = r;
        self
    }
    pub fn set_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }
    pub fn build(self) -> NeuralNet {
        NeuralNet::new(
            self.layer_sizes,
            self.afs,
            self.learning_rate,
            self.batch_size,
        )
    }
}

#[derive(Clone, Debug)]
pub struct NeuralNet {
    biases: Vec<Matrix>,
    weights: Vec<Matrix>,
    afs: Vec<ActivationFunction>,
    layer_sizes: Vec<usize>,

    // hyper-parameters
    learning_rate: f64,
    batch_size: usize,
}

impl NeuralNet {
    pub fn new(
        layer_sizes: Vec<usize>,
        afs: Vec<ActivationFunction>,
        learning_rate: f64,
        batch_size: usize,
    ) -> Self {
        assert_eq!(afs.len(), layer_sizes.len() - 2);
        assert!(0.0 < learning_rate && learning_rate <= 1.0);

        Self {
            biases: layer_sizes
                .iter()
                .skip(1)
                .map(|&n| Matrix::new_random(1, n))
                .collect(),
            weights: layer_sizes
                .array_windows()
                .map(|&[m, n]| Matrix::new_random(m, n))
                .collect(),
            afs,
            layer_sizes,
            learning_rate,
            batch_size,
        }
    }

    pub fn forward(&self, mut xs: Matrix) -> Matrix {
        assert_eq!(xs.dimensions(), (1, self.layer_sizes[0]));

        for l in 0..self.depth() {
            xs = &xs * &self.weights[l];
            xs += &self.biases[l];
            xs = if l == self.depth() - 1 {
                // softmax in output layer
                xs = xs.map(|x| x.exp());
                let sum = xs.reduce(|a, b| a + b, 0.0);
                xs.map(|x| x / sum)
            } else {
                self.afs[l].map(&xs)
            };
        }
        xs
    }

    pub fn accuracy(&self, dataset: &Dataset) -> f64 {
        self.assert_matching_dataset(dataset);

        let mut counter = 0;
        for (input, label) in dataset.data() {
            let (i, _) = self.forward(Matrix::new_row(input.clone())).data()[0]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();
            if i == *label {
                counter += 1;
            }
        }
        counter as f64 / dataset.data().len() as f64
    }

    pub fn cross_entropy(&self, dataset: &Dataset) -> f64 {
        let start = Instant::now();

        self.assert_matching_dataset(dataset);

        let mut res = 0.0;
        for (input, label) in dataset.data() {
            let predicted = self.forward(Matrix::new_row(input.clone()));
            // -log(p_label)
            res += -predicted.data()[0][*label].ln();
        }

        println!("loss took {:?}", start.elapsed());
        res / dataset.data().len() as f64
    }
    pub fn train(&mut self, dataset: &mut Dataset, epochs: usize) {
        self.assert_matching_dataset(dataset);

        for _ in 0..epochs {
            let start = Instant::now();

            dataset.shuffle();

            let mut counter = 0;
            let mut weight_gradient = self
                .layer_sizes
                .array_windows()
                .map(|&[m, n]| Matrix::new_zero(m, n))
                .collect::<Vec<Matrix>>();
            let mut bias_gradient = self
                .layer_sizes
                .iter()
                .skip(1)
                .map(|&n| Matrix::new_row(vec![0.0; n]))
                .collect::<Vec<Matrix>>();

            let points = dataset.data().len();

            for i in 0..points {
                let (input, label) = dataset.data()[i].clone();

                // in the beginning we have an 'a'
                // layer = (w, z, a)
                let mut a = vec![Matrix::new_row(input)];
                let mut z = vec![];
                for l in 0..self.depth() {
                    let mut z_curr = a.last().unwrap() * &self.weights[l];
                    z_curr += &self.biases[l];
                    z.push(z_curr);
                    if l == self.depth() - 1 {
                        // softmax
                        let ps = z.last().unwrap().map(|x| x.exp());
                        let sum = ps.reduce(|a, b| a + b, 0.0);
                        a.push(ps.map(|x| x / sum));
                    } else {
                        a.push(self.afs[l].map(z.last().unwrap()));
                    }
                }

                // dL/dz = ŷ - y
                let mut deltas = Matrix::new_row(vec![0.0; dataset.dimensions().1]);
                for i in 0..dataset.dimensions().1 {
                    deltas.data_mut()[0][i] =
                        a.last().unwrap().data()[0][i] - (i == label) as u8 as f64
                }

                for l in (0..self.depth()).rev() {
                    // delta is the loss w.r.t. the last activations
                    // make delta the loss w.r.t. the weighted sum z by multiplying with f'(z)
                    if l < self.depth() - 1 {
                        deltas = deltas.element_mul(&self.afs[l].map_derivative(&z[l]));
                    }
                    bias_gradient[l] += &deltas;
                    weight_gradient[l] += &(&a[l].transposed() * &deltas);
                    deltas = &deltas * &self.weights[l].transposed();
                }

                counter += 1;
                if counter == self.batch_size || i == points - 1 {
                    for l in 0..self.depth() {
                        // update parameters
                        let f = self.learning_rate * 1.0 / counter as f64;
                        self.weights[l] -= &(&weight_gradient[l] * f);
                        self.biases[l] -= &(&bias_gradient[l] * f);

                        // reset gradients back to 0
                        weight_gradient[l] =
                            Matrix::new_zero(self.layer_sizes[l], self.layer_sizes[l + 1]);
                        bias_gradient[l] = Matrix::new_row(vec![0.0; self.layer_sizes[l + 1]]);
                    }
                    counter = 0;
                }
            }
            println!("one epoch took {:?}", start.elapsed());
        }
    }

    pub fn assert_matching_dataset(&self, dataset: &Dataset) {
        assert_eq!(dataset.dimensions(), self.dimensions());
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.layer_sizes[0], *self.layer_sizes.last().unwrap())
    }

    // # of hidden layers
    pub fn depth(&self) -> usize {
        self.layer_sizes.len() - 1
    }
}

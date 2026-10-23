use rand::seq::SliceRandom;

use crate::linalg::Matrix;

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

pub struct Dataset(pub Vec<(Vec<f64>, Vec<f64>)>);
impl Dataset {
    pub fn new(mut data: Vec<(Vec<f64>, Vec<f64>)>) -> Self {
        data.shuffle(&mut rand::rng());
        for [(i1, o1), (i2, o2)] in data.array_windows() {
            assert_eq!(i1.len(), i2.len());
            assert_eq!(o1.len(), o2.len());
        }
        Self(data)
    }
    /// 80/20
    pub fn split(self) -> (Self, Self) {
        let mut a = self.0;
        let b = a.split_off(a.len() * 4 / 5);
        (Self::new(a), Self::new(b))
    }
}

pub struct NeuralNetBuilder {
    afs: Vec<ActivationFunction>,
    learning_rate: f64,
    layer_sizes: Vec<usize>,
}
impl NeuralNetBuilder {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        Self {
            afs: vec![ActivationFunction::Sigmoid; layer_sizes.len() - 1],
            learning_rate: 0.01,
            layer_sizes,
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
    pub fn build(self) -> NeuralNet {
        NeuralNet::new(self.layer_sizes, self.afs, self.learning_rate)
    }
}

pub struct NeuralNet {
    biases: Vec<Matrix>,
    weights: Vec<Matrix>,
    afs: Vec<ActivationFunction>,
    layer_sizes: Vec<usize>,

    // hyper-parameters
    learning_rate: f64,
}

impl NeuralNet {
    pub fn new(layer_sizes: Vec<usize>, afs: Vec<ActivationFunction>, learning_rate: f64) -> Self {
        assert_eq!(afs.len(), layer_sizes.len() - 1);
        assert!(0.0 < learning_rate && learning_rate <= 1.0);

        Self {
            biases: layer_sizes
                .iter()
                .skip(1)
                .map(|&n| Matrix::new_random(n, 1))
                .collect(),
            weights: layer_sizes
                .array_windows()
                .map(|&[m, n]| Matrix::new_random(m, n))
                .collect(),
            afs,
            layer_sizes,
            learning_rate,
        }
    }

    pub fn forward(&self, mut xs: Matrix) -> Matrix {
        assert_eq!(xs.dimensions(), (self.layer_sizes[0], 1));
        for l in 0..self.depth() {
            xs = self.afs[l].map(&(&(&self.weights[l].transposed() * &xs) + &self.biases[l]));
        }
        xs
    }

    pub fn mse(&self, dataset: &Dataset) -> f64 {
        let mut res = 0.0;
        for (input, target) in dataset.0.clone() {
            let diff = &Matrix::new_column(target) - &self.forward(Matrix::new_column(input));
            res += diff.reduce(|acc, x| acc + x * x, 0.0);
        }
        res / dataset.0.len() as f64
    }
    pub fn train(&mut self, dataset: &Dataset, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in dataset.0.clone() {
                self.sgd(Matrix::new_column(input), Matrix::new_column(target));
            }
        }
    }

    fn sgd(&mut self, input: Matrix, target: Matrix) {
        assert_eq!(input.dimensions(), (self.layer_sizes[0], 1));
        assert_eq!(target.dimensions(), (*self.layer_sizes.last().unwrap(), 1));

        // in the beginning we have an 'a'
        // layer = (w, z, a)
        let mut a = vec![input.clone()];
        let mut z = vec![];
        for l in 0..self.depth() {
            z.push(&(&self.weights[l].transposed() * a.last().unwrap()) + &self.biases[l]);
            a.push(self.afs[l].map(z.last().unwrap()));
        }

        // ŷ - y
        let mut deltas = &(a.last().unwrap() - &target) * 2.0;

        for l in (0..self.depth()).rev() {
            // delta is the loss w.r.t. the last activations
            // make delta the loss w.r.t. the weighted sum z by multiplying with f'(z)
            deltas = deltas.element_mul(&self.afs[l].map_derivative(&z[l]));
            // update bias, bias directly affects z
            self.biases[l] -= &(&deltas * self.learning_rate);
            // update weights
            self.weights[l] -= &(&(&a[l] * &deltas.transposed()) * self.learning_rate);
            // update deltas
            deltas = &self.weights[l] * &deltas;
        }
    }

    pub fn depth(&self) -> usize {
        self.layer_sizes.len() - 1
    }
}

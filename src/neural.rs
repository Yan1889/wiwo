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
                .map(|&[m, n]| Matrix::new_random(n, m))
                .collect(),
            afs,
            layer_sizes,
            learning_rate,
        }
    }

    pub fn forward(&self, mut xs: Matrix) -> Matrix {
        assert_eq!(xs.dimensions(), (self.layer_sizes[0], 1));
        for l in 0..self.depth() {
            xs = (&(&self.weights[l] * &xs) + &self.biases[l]).map(|x| self.afs[l].apply(x));
        }
        xs
    }

    pub fn train(&mut self, input: Matrix, target: Matrix) {
        assert_eq!(input.dimensions(), (self.layer_sizes[0], 1));
        assert_eq!(target.dimensions(), (*self.layer_sizes.last().unwrap(), 1));

        let mut a = vec![input.clone()];
        let mut z = vec![input];
        for l in 0..self.depth() {
            z.push(&(&self.weights[l] * a.last().unwrap()) + &self.biases[l]);
            a.push(z.last().unwrap().map(|x| self.afs[l].apply(x)));
        }

        // ŷ - y
        let mut deltas = (a.last().unwrap() - &target).map(|x| 2.0 * x);

        for l in (0..self.depth()).rev() {
            // delta is the loss w.r.t. the last activations
            // make delta the loss w.r.t. the weighted sum z by multiplying with f'(z)
            deltas = deltas.element_mul(&z[l + 1].map(|x| self.afs[l].derivative(x)));
            // update bias, bias directly affects z
            self.biases[l] -= &deltas.map(|x| x * self.learning_rate);
            // update weights
            self.weights[l] -= &(&a[l] * &deltas.transpose()).transpose();
            // update deltas
            deltas = (&deltas.transpose() * &self.weights[l]).transpose();
        }
    }

    pub fn depth(&self) -> usize {
        self.layer_sizes.len() - 1
    }
}

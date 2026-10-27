use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use rand::RngExt;

#[derive(Clone, Debug)]
pub struct Matrix(Vec<Vec<f64>>);

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        for [a, b] in data.array_windows() {
            assert_eq!(a.len(), b.len());
        }
        Self(data)
    }
    pub fn new_zero(m: usize, n: usize) -> Self {
        Self(vec![vec![0.0; n]; m])
    }
    pub fn new_random(m: usize, n: usize) -> Self {
        let mut res = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                res[i][j] = rand::rng().random_range(-1.0..1.0);
            }
        }
        Self(res)
    }
    pub fn new_row(xs: Vec<f64>) -> Self {
        Self(vec![xs])
    }
    pub fn dimensions(&self) -> (usize, usize) {
        (self.0.len(), self.0[0].len())
    }
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        let (m, n) = self.dimensions();
        let mut res = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                res[i][j] = f(self.0[i][j]);
            }
        }
        Self(res)
    }
    pub fn transposed(&self) -> Self {
        let (m, n) = self.dimensions();
        let mut res = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                res[i][j] = self.0[j][i];
            }
        }
        Self(res)
    }
    pub fn element_mul(&self, other: &Matrix) -> Self {
        assert_eq!(self.dimensions(), other.dimensions());
        let (m, n) = self.dimensions();
        let mut res = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                res[i][j] = self.0[i][j] * other.0[i][j];
            }
        }
        Self(res)
    }
    pub fn reduce<T>(&self, f: impl Fn(T, f64) -> T, mut acc: T) -> T {
        let (m, n) = self.dimensions();
        for i in 0..m {
            for j in 0..n {
                acc = f(acc, self.0[i][j]);
            }
        }
        acc
    }
    pub fn data(&self) -> &[Vec<f64>] {
        &self.0
    }
    pub fn data_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.0
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Self::Output {
        assert_eq!(self.dimensions(), other.dimensions());
        let (m, n) = self.dimensions();
        let mut res = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                res[i][j] = self.0[i][j] + other.0[i][j];
            }
        }
        Matrix(res)
    }
}
impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, other: &Matrix) {
        assert_eq!(self.dimensions(), other.dimensions());
        let (m, n) = self.dimensions();
        for i in 0..m {
            for j in 0..n {
                self.data_mut()[i][j] += other.data()[i][j];
            }
        }
    }
}
impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, other: &Matrix) -> Self::Output {
        assert_eq!(self.dimensions(), other.dimensions());
        let (m, n) = self.dimensions();
        let mut res = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                res[i][j] = self.0[i][j] - other.0[i][j];
            }
        }
        Matrix(res)
    }
}
impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, other: &Matrix) {
        assert_eq!(self.dimensions(), other.dimensions());
        let (m, n) = self.dimensions();
        for i in 0..m {
            for j in 0..n {
                self.data_mut()[i][j] -= other.data()[i][j];
            }
        }
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, other: &Matrix) -> Self::Output {
        let (m, self_n) = self.dimensions();
        let (other_m, n) = other.dimensions();
        assert_eq!(self_n, other_m);

        let mut res = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                for k in 0..self_n {
                    res[i][j] += self.0[i][k] * other.0[k][j];
                }
            }
        }
        Matrix(res)
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;
    fn mul(self, x: f64) -> Self::Output {
        let (m, n) = self.dimensions();
        let mut res = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                res[i][j] = self.0[i][j] * x;
            }
        }
        Matrix(res)
    }
}

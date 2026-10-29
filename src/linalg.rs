use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use rand::RngExt;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        Self { rows, cols, data }
    }
    pub fn new_zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    pub fn new_random(rows: usize, cols: usize) -> Self {
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows * cols {
            data[i] = rand::rng().random_range(-0.1..0.1);
        }
        Self { rows, cols, data }
    }
    pub fn new_row(xs: Vec<f64>) -> Self {
        Self {
            rows: 1,
            cols: xs.len(),
            data: xs,
        }
    }
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows * self.cols {
            data[i] = f(self.data[i]);
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
    pub fn transposed(&self) -> Self {
        let mut res = Matrix::new_zero(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set(j, i, self.get(i, j));
            }
        }
        res
    }
    pub fn element_mul(&self, other: &Matrix) -> Self {
        assert_eq!(self.dimensions(), other.dimensions());
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows * self.cols {
            data[i] = self.data[i] * other.data[i];
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
    pub fn reduce<T>(&self, f: impl Fn(T, f64) -> T, mut acc: T) -> T {
        for i in 0..self.rows * self.cols {
            acc = f(acc, self.data[i]);
        }
        acc
    }
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn set(&mut self, i: usize, j: usize, x: f64) {
        self.data[i * self.cols + j] = x;
    }
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, other: &Matrix) -> Self::Output {
        assert_eq!(self.dimensions(), other.dimensions());
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows * self.cols {
            data[i] = self.data[i] + other.data[i];
        }
        Matrix {
            cols: self.cols,
            rows: self.rows,
            data,
        }
    }
}
impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, other: &Matrix) {
        assert_eq!(self.dimensions(), other.dimensions());
        for i in 0..self.rows * self.cols {
            self.data[i] += other.data[i];
        }
    }
}
impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, other: &Matrix) -> Self::Output {
        assert_eq!(self.dimensions(), other.dimensions());
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows * self.cols {
            data[i] = self.data[i] - other.data[i];
        }
        Matrix {
            cols: self.cols,
            rows: self.rows,
            data,
        }
    }
}
impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, other: &Matrix) {
        assert_eq!(self.dimensions(), other.dimensions());
        for i in 0..self.rows * self.cols {
            self.data[i] -= other.data[i];
        }
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, other: &Matrix) -> Self::Output {
        assert_eq!(self.cols, other.rows);
        let mut res = Matrix::new_zero(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    *res.get_mut(i, j) += self.get(i, k) * other.get(k, j);
                }
            }
        }
        res
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;
    fn mul(self, x: f64) -> Self::Output {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows * self.cols {
            data[i] = self.data[i] * x;
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, x: f64) {
        for i in 0..self.rows * self.cols {
            self.data[i] *= x;
        }
    }
}

impl Div<f64> for &Matrix {
    type Output = Matrix;
    fn div(self, x: f64) -> Self::Output {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows * self.cols {
            data[i] = self.data[i] / x;
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl DivAssign<f64> for Matrix {
    fn div_assign(&mut self, x: f64) {
        *self *= 1.0 / x;
    }
}

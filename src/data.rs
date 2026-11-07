use std::{
    f64::consts::{PI, TAU},
    fs::File,
    io::{BufReader, Read},
    ops::Range,
};

use ordered_float::OrderedFloat;
use rand::{random, random_range, seq::SliceRandom};

pub struct Dataset {
    pub data: Vec<(Vec<f64>, usize)>,
    pub classes: usize,
}

impl Dataset {
    pub fn new(mut data: Vec<(Vec<f64>, usize)>, classes: usize) -> Self {
        for [(i1, l1), (i2, l2)] in data.array_windows() {
            assert_eq!(i1.len(), i2.len());
            assert!(*l1 < classes && *l2 < classes);
        }
        data.shuffle(&mut rand::rng());
        Self { data, classes }
    }

    pub fn new_parametric(
        equations: Vec<(fn(f64) -> Vec<f64>, Range<f64>)>,
        noise: f64,
        samples: usize,
    ) -> Self {
        let n = equations.len();
        let mut data = vec![];
        for (i, (f, r)) in equations.into_iter().enumerate() {
            let mut t = r.start;
            while t < r.end {
                let mut point = f(t);
                for x in &mut point {
                    *x += rand::random_range(-noise..=noise);
                }
                data.push((point, i));
                t += (r.end - r.start) / samples as f64;
            }
        }
        Self::new(data, n)
    }

    pub fn normalized(mut self) -> Self {
        let (min, max) = self
            .data
            .iter()
            .flat_map(|(xs, _)| xs.iter().enumerate())
            .fold(
                (
                    vec![f64::MAX; self.dimensions().0],
                    vec![f64::MIN; self.dimensions().0],
                ),
                |(mut min, mut max), (i, &x)| {
                    min[i] = min[i].min(x);
                    max[i] = max[i].max(x);
                    (min, max)
                },
            );
        for (input, _) in &mut self.data {
            for (i, x) in input.iter_mut().enumerate() {
                *x = (*x - min[i]) / (max[i] - min[i]);
            }
        }
        self
    }

    pub fn get_titanic() -> Self {
        let csv = std::fs::read_to_string("data/titanic/train.csv").unwrap();
        let mut data = Vec::<(Vec<f64>, usize)>::new();
        for l in csv.lines().skip(1) {
            let row = l.split(',').collect::<Vec<&str>>();
            let label = row[1].parse::<usize>().unwrap();
            let (class1, class2, class3) = Self::parse_class(row[2]);
            let gender = Self::parse_gender(row[5]);
            let age = Self::parse_age(row[6]);
            let sib_sp = row[7].parse::<f64>().unwrap();
            let parch = row[8].parse::<f64>().unwrap();
            let fare = Self::parse_fare(row[10]);
            let (embarked_c, embarked_q, embarked_s) = Self::parse_embarked(row[12]);
            let input = vec![
                class1, class2, class3, gender, age, sib_sp, parch, fare, embarked_c, embarked_q,
                embarked_s,
            ];
            data.push((input, label));
        }
        Self::new(data, 2)
    }

    pub fn get_titanic_test() -> Vec<(usize, Vec<f64>)> {
        let mut res = vec![];
        let csv = std::fs::read_to_string("data/titanic/test.csv").unwrap();
        for l in csv.lines().skip(1) {
            let row = l.split(',').collect::<Vec<&str>>();
            let id = row[0].parse::<usize>().unwrap();
            let (class1, class2, class3) = Self::parse_class(row[1]);
            let gender = Self::parse_gender(row[4]);
            let age = Self::parse_age(row[5]);
            let sib_sp = row[6].parse::<f64>().unwrap();
            let parch = row[7].parse::<f64>().unwrap();
            let fare = Self::parse_fare(row[9]);
            let (embarked_c, embarked_q, embarked_s) = Self::parse_embarked(row[11]);
            let input = vec![
                class1, class2, class3, gender, age, sib_sp, parch, fare, embarked_c, embarked_q,
                embarked_s,
            ];
            res.push((id, input));
        }
        res
    }
    fn parse_class(x: &str) -> (f64, f64, f64) {
        match x {
            "1" => (1.0, 0.0, 0.0),
            "2" => (0.0, 1.0, 0.0),
            "3" => (0.0, 0.0, 1.0),
            _ => unreachable!(),
        }
    }
    fn parse_age(x: &str) -> f64 {
        x.parse::<f64>().unwrap_or(30.0) / 100.0
    }
    fn parse_gender(x: &str) -> f64 {
        (x == "male") as u8 as f64
    }
    fn parse_fare(x: &str) -> f64 {
        (1.0 + x.parse::<f64>().unwrap_or(0.0)).ln() / 6.0
    }
    fn parse_embarked(x: &str) -> (f64, f64, f64) {
        match x {
            "C" => (1.0, 0.0, 0.0),
            "Q" => (0.0, 1.0, 0.0),
            "S" => (0.0, 0.0, 1.0),
            "" => (0.0, 0.0, 0.0),
            _ => unreachable!(),
        }
    }

    /// first column is label, the rest are inputs in [0; 1]
    pub fn load_csv_train(filename: &str) -> Self {
        let csv = std::fs::read_to_string(filename).unwrap();
        let mut data = Vec::new();
        let mut max_label = 0;
        for l in csv.lines().skip(1) {
            let mut it = l.split(',');
            let label = it.next().unwrap().parse::<usize>().unwrap();
            max_label = max_label.max(label);
            data.push((it.map(|x| x.parse::<f64>().unwrap()).collect(), label));
        }
        Self::new(data, max_label + 1)
    }

    /// first column is id, the rest are inputs in [0; 1]
    pub fn load_csv_test(filename: &str) -> Vec<(usize, Vec<f64>)> {
        let csv = std::fs::read_to_string(filename).unwrap();
        let mut data = Vec::new();
        for l in csv.lines().skip(1) {
            let mut it = l.split(',');
            let id = it.next().unwrap().parse::<usize>().unwrap();
            data.push((id, it.map(|x| x.parse::<f64>().unwrap()).collect()));
        }
        data
    }

    /// 4D -> {0, 1, 2, 3}
    pub fn get_iris() -> Self {
        let csv = std::fs::read_to_string("data/iris.csv").unwrap();
        let mut data = Vec::<(Vec<f64>, usize)>::new();
        for l in csv.lines().skip(1) {
            let row = l.split(',').skip(1).collect::<Vec<&str>>();
            let input = (0..4)
                .map(|i| row[i].parse::<f64>().unwrap())
                .collect::<Vec<f64>>();
            let label = match row[4] {
                "Iris-setosa" => 0,
                "Iris-versicolor" => 1,
                "Iris-virginica" => 2,
                _ => unreachable!(),
            };
            data.push((input, label));
        }
        Self::new(data, 3)
    }

    pub fn get_iris_2d() -> Self {
        let mut full = Self::get_iris();
        for (input, _) in &mut full.data {
            *input = vec![input[2], input[3]];
        }
        full
    }

    pub fn load_mnist_file(file_images: &str, file_labels: &str) -> Self {
        let mut data = Vec::new();

        let mut br_image = BufReader::new(File::open(file_images).unwrap());
        let mut br_label = BufReader::new(File::open(file_labels).unwrap());

        let mut magic = [0; 4];
        br_image.read_exact(&mut magic).unwrap();
        assert_eq!(0x00000803, u32::from_be_bytes(magic));
        br_label.read_exact(&mut magic).unwrap();
        assert_eq!(0x00000801, u32::from_be_bytes(magic));

        let mut count_images = [0; 4];
        br_image.read_exact(&mut count_images).unwrap();
        let mut count_labels = [0; 4];
        br_label.read_exact(&mut count_labels).unwrap();
        assert_eq!(count_images, count_labels);

        let mut rows = [0; 4];
        br_image.read_exact(&mut rows).unwrap();
        assert_eq!(u32::from_be_bytes(rows), 28);
        let mut cols = [0; 4];
        br_image.read_exact(&mut cols).unwrap();
        assert_eq!(u32::from_be_bytes(cols), 28);

        // for _ in 0..u32::from_be_bytes(count) {
        for _ in 0..u32::from_be_bytes(count_images) {
            let mut buf = vec![0; 784];
            br_image.read_exact(&mut buf).unwrap();
            let mut label = [0];
            br_label.read_exact(&mut label).unwrap();
            data.push((
                buf.into_iter().map(|x| x as f64 / 255.0).collect(),
                label[0] as usize,
            ));
        }
        Self::new(data, 10)
    }

    /// angle (in radians), scale and translation
    pub fn transform_digit(digit: &mut [f64], a: f64, s: f64, (x_t, y_t): (f64, f64)) {
        // input validation
        assert_eq!(digit.len(), 784);

        let mut new = vec![0.0; 784];

        for row_dst in 0..28 {
            for col_dst in 0..28 {
                let i_dst = row_dst * 28 + col_dst;

                // first translate to center
                let mut x_src = col_dst as f64 - 13.5;
                let mut y_src = row_dst as f64 - 13.5;
                // then translate (inverted)
                (x_src, y_src) = (x_src - x_t, y_src - y_t);
                // then scale (inverted)
                (x_src, y_src) = (x_src / s, y_src / s);
                // then rotate (inverted)
                (x_src, y_src) = (
                    x_src * a.cos() + y_src * a.sin(),
                    -x_src * a.sin() + y_src * a.cos(),
                );
                // then translate back
                (x_src, y_src) = (x_src + 13.5, y_src + 13.5);

                // linear interpolation:
                let x0 = x_src.floor();
                let y0 = y_src.floor();
                let fx = x_src - x0;
                let fy = y_src - y0;

                let neighbors = [
                    (y0, x0),
                    (y0, x0 + 1.0),
                    (y0 + 1.0, x0),
                    (y0 + 1.0, x0 + 1.0),
                ];
                let weights = [
                    (1.0 - fx) * (1.0 - fy),
                    fx * (1.0 - fy),
                    (1.0 - fx) * fy,
                    fx * fy,
                ];
                for i in 0..4 {
                    let (row_src, col_src) = neighbors[i];
                    let row_src = row_src as i32;
                    let col_src = col_src as i32;
                    if (0..28).contains(&row_src) && (0..28).contains(&col_src) {
                        let i_src = row_src as usize * 28 + col_src as usize;
                        // indexing is safe
                        new[i_dst] += digit[i_src] * weights[i];
                    }
                }
                // a little noise
                new[i_dst] += rand::random_range(0.0..0.1);
            }
        }
        digit.copy_from_slice(&new);
    }

    pub fn randomize_mnist(&mut self) {
        assert_eq!(self.dimensions(), (784, 10));
        for (digit, _) in &mut self.data {
            // 10° random angle
            let angle = rand::random_range(-PI / 18.0..PI / 18.0);
            // random scale
            let scale = rand::random_range(0.8..1.2);
            // 5px random shift
            let translation = (rand::random_range(-3.0..3.0), rand::random_range(-3.0..3.0));
            Self::transform_digit(digit, angle, scale, translation);
        }
    }

    pub fn get_mnist() -> (Self, Self) {
        (
            Self::load_mnist_file(
                "data/mnist/train-images.idx3-ubyte",
                "data/mnist/train-labels.idx1-ubyte",
            ),
            Self::load_mnist_file(
                "data/mnist/t10k-images.idx3-ubyte",
                "data/mnist/t10k-labels.idx1-ubyte",
            ),
        )
    }

    pub fn get_fashion_mnist() -> (Self, Self) {
        (
            Self::load_mnist_file(
                "data/fashion-mnist/train-images.idx3-ubyte",
                "data/fashion-mnist/train-labels.idx1-ubyte",
            ),
            Self::load_mnist_file(
                "data/fashion-mnist/t10k-images.idx3-ubyte",
                "data/fashion-mnist/t10k-labels.idx1-ubyte",
            ),
        )
    }

    /// 2D -> [0; 1]
    pub fn get_heart() -> Self {
        Self::new_parametric(
            vec![
                (
                    |t| {
                        vec![
                            t.sin().powi(3) / 3.0 + 0.5,
                            (13.0 * t.cos()
                                - 5.0 * (2.0 * t).cos()
                                - 2.0 * (3.0 * t).cos()
                                - (4.0 * t).cos())
                                / 48.0
                                + 0.5,
                        ]
                    },
                    0.0..TAU,
                ),
                (
                    |t| {
                        vec![
                            t.sin().powi(3) / 2.0 + 0.5,
                            (13.0 * t.cos()
                                - 5.0 * (2.0 * t).cos()
                                - 2.0 * (3.0 * t).cos()
                                - (4.0 * t).cos())
                                / (16.0 * 2.0)
                                + 0.55,
                        ]
                    },
                    0.0..TAU,
                ),
                (
                    |t| {
                        vec![
                            t.sin().powi(3) / 6.0 + 0.5,
                            (13.0 * t.cos()
                                - 5.0 * (2.0 * t).cos()
                                - 2.0 * (3.0 * t).cos()
                                - (4.0 * t).cos())
                                / (16.0 * 6.0)
                                + 0.45,
                        ]
                    },
                    0.0..TAU,
                ),
            ],
            0.00,
            100,
        )
    }

    pub fn get_moon() -> Self {
        Self::new_parametric(
            vec![
                (
                    |t| vec![(t.cos() + 1.0) / 3.0, (t.sin() + 1.0) / 3.0],
                    0.0..PI,
                ),
                (
                    |t| vec![(t.cos() + 2.0) / 3.0, (-t.sin() + 1.5) / 3.0],
                    0.0..PI,
                ),
                (|t| vec![t, 1.3 - t], 0.5..0.8),
            ],
            0.01,
            100,
        )
    }

    /// 2D -> binary label
    pub fn get_circular() -> Self {
        Self::new_parametric(
            vec![
                (|t| vec![t.sin() / 3.0 + 0.5, t.cos() / 3.0 + 0.5], 0.0..TAU),
                (|t| vec![t.sin() / 6.0 + 0.5, t.cos() / 6.0 + 0.5], 0.0..TAU),
            ],
            0.03,
            50,
        )
    }

    /// 2D -> binary label
    pub fn get_spiral() -> Self {
        Self::new_parametric(
            vec![
                (
                    |t| vec![0.05 * t * t.cos() + 0.5, 0.05 * t * t.sin() + 0.5],
                    0.0..10.0,
                ),
                (
                    |t| vec![-0.05 * t * t.cos() + 0.5, -0.05 * t * t.sin() + 0.5],
                    0.0..10.0,
                ),
            ],
            0.01,
            100,
        )
    }
    /// 2D -> binary label
    pub fn get_xor() -> Self {
        let samples = 300;
        Self::new(
            (0..samples)
                .map(|i| {
                    // 0..25 => (0, 0)
                    // 25..50 => (1, 0)
                    // 50..75 => (0, 1)
                    // 75..100 => (1, 1)
                    (
                        vec![
                            rand::random_range(
                                if i < samples / 4 || (samples / 2..samples * 3 / 4).contains(&i) {
                                    0.0..0.5
                                } else {
                                    0.5..1.0
                                },
                            ),
                            rand::random_range(if i < samples / 2 { 0.0..0.5 } else { 0.5..1.0 }),
                        ],
                        ((samples / 4..samples * 3 / 4).contains(&i)) as usize,
                    )
                })
                .collect(),
            2,
        )
    }

    pub fn get_cross() -> Self {
        Self::new_parametric(
            vec![(|t| vec![0.5, t], 0.2..0.8), (|t| vec![t, 0.5], 0.2..0.8)],
            0.01,
            100,
        )
    }

    pub fn get_triple() -> Self {
        // at (0.2, 0.2) (0.4, 0.8) (0.8, 0.6)
        let mut data = Vec::new();
        for (i, &(x, y)) in [(0.2, 0.2), (0.4, 0.8), (0.8, 0.6)].iter().enumerate() {
            for _ in 0..50 {
                data.push((
                    vec![
                        x + rand::random_range(-0.1..0.1),
                        y + rand::random_range(-0.1..0.1),
                    ],
                    i,
                ));
            }
        }
        Self::new(data, 3)
    }

    /*
    pub fn knn_sort(&mut self, p: &[f64]) {
        self.data.sort_by_key(|(a, _)| {
            OrderedFloat(
                a.iter()
                    .zip(p.iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum::<f64>(),
            )
        });
    }*/

    /// returns 75% train, 25% test
    pub fn split(self) -> (Self, Self) {
        let mut a = self.data;
        let b = a.split_off(a.len() * 3 / 4);
        (Self::new(a, self.classes), Self::new(b, self.classes))
    }

    pub fn shuffle(&mut self) {
        self.data.shuffle(&mut rand::rng());
    }

    /// access to internal data via immutable slice
    pub fn data(&self) -> &[(Vec<f64>, usize)] {
        &self.data
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.data[0].0.len(), self.classes)
    }
}

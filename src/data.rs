use std::{
    f64::consts::TAU,
    fs::File,
    io::{BufReader, Read},
    ops::Range,
};

use rand::seq::SliceRandom;

pub struct Dataset {
    data: Vec<(Vec<f64>, usize)>,
    classes: usize,
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
        a: fn(f64) -> Vec<f64>,
        r1: Range<f64>,
        b: fn(f64) -> Vec<f64>,
        r2: Range<f64>,
        noise: f64,
        samples: usize,
    ) -> Self {
        let mut data = vec![];
        let mut t = r1.start;
        while t <= r1.end {
            let mut point = a(t);
            for x in &mut point {
                *x += rand::random_range(-noise..=noise);
            }
            data.push((point, 0));
            t += (r1.end - r1.start) / samples as f64;
        }
        t = r2.start;
        while t <= r2.end {
            let mut point = b(t);
            for x in &mut point {
                *x += rand::random_range(-noise..=noise);
            }
            data.push((point, 1));
            t += (r2.end - r2.start) / samples as f64;
        }
        Self::new(data, 2)
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

    pub fn get_mnist() -> Self {
        let mut data = Vec::new();

        let mut br_image = BufReader::new(File::open("data/train-images.idx3-ubyte").unwrap());
        let mut br_label = BufReader::new(File::open("data/train-labels.idx1-ubyte").unwrap());

        let mut magic = [0; 4];
        br_image.read_exact(&mut magic).unwrap();
        assert_eq!(0x00000803, u32::from_be_bytes(magic));
        br_label.read_exact(&mut magic).unwrap();
        assert_eq!(0x00000801, u32::from_be_bytes(magic));

        let mut count = [0; 4];
        br_image.read_exact(&mut count).unwrap();
        assert_eq!(u32::from_be_bytes(count), 60_000);
        br_label.read_exact(&mut count).unwrap();
        assert_eq!(u32::from_be_bytes(count), 60_000);

        let mut rows = [0; 4];
        br_image.read_exact(&mut rows).unwrap();
        assert_eq!(u32::from_be_bytes(rows), 28);
        let mut cols = [0; 4];
        br_image.read_exact(&mut cols).unwrap();
        assert_eq!(u32::from_be_bytes(cols), 28);

        // for _ in 0..u32::from_be_bytes(count) {
        for _ in 0..60_000 {
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

    /// 2D -> [0; 1]
    pub fn get_heart() -> Self {
        Self::new_parametric(
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
            |t| {
                if t < TAU {
                    vec![
                        t.sin().powi(3) / 2.0 + 0.5,
                        (13.0 * t.cos()
                            - 5.0 * (2.0 * t).cos()
                            - 2.0 * (3.0 * t).cos()
                            - (4.0 * t).cos())
                            / (16.0 * 2.0)
                            + 0.55,
                    ]
                } else {
                    vec![
                        t.sin().powi(3) / 6.0 + 0.5,
                        (13.0 * t.cos()
                            - 5.0 * (2.0 * t).cos()
                            - 2.0 * (3.0 * t).cos()
                            - (4.0 * t).cos())
                            / (16.0 * 6.0)
                            + 0.45,
                    ]
                }
            },
            0.0..2.0 * TAU,
            0.00,
            500,
        )
    }

    /// 2D -> binary label
    pub fn get_circular() -> Self {
        Self::new_parametric(
            |t| vec![t.sin() / 3.0 + 0.5, t.cos() / 3.0 + 0.5],
            0.0..TAU,
            |t| vec![t.sin() / 6.0 + 0.5, t.cos() / 6.0 + 0.5],
            0.0..TAU,
            0.03,
            100,
        )
    }

    /// 2D -> binary label
    pub fn get_spiral() -> Self {
        Self::new_parametric(
            |t| vec![0.05 * t * t.cos() + 0.5, 0.05 * t * t.sin() + 0.5],
            0.0..10.0,
            |t| vec![-0.05 * t * t.cos() + 0.5, -0.05 * t * t.sin() + 0.5],
            0.0..10.0,
            0.01,
            1000,
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
            |t| vec![0.5, t],
            0.2..0.8,
            |t| vec![t, 0.5],
            0.2..0.8,
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

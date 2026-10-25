use std::{f64::consts::TAU, ops::Range};

use rand::seq::SliceRandom;

pub struct Dataset(Vec<(Vec<f64>, Vec<f64>)>);
impl Dataset {
    pub fn new(mut data: Vec<(Vec<f64>, Vec<f64>)>) -> Self {
        for [(i1, o1), (i2, o2)] in data.array_windows() {
            assert_eq!(i1.len(), i2.len());
            assert_eq!(o1.len(), o2.len());
        }
        data.shuffle(&mut rand::rng());
        Self(data)
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
            data.push((point, vec![0.0]));
            t += (r1.end - r1.start) / samples as f64;
        }
        t = r2.start;
        while t <= r2.end {
            let mut point = b(t);
            for x in &mut point {
                *x += rand::random_range(-noise..=noise);
            }
            data.push((point, vec![1.0]));
            t += (r2.end - r2.start) / samples as f64;
        }
        Self::new(data)
    }

    /// 4D -> {0, 1, 2, 3}
    pub fn get_iris() -> Self {
        let csv = std::fs::read_to_string("data/iris.csv").unwrap();
        let mut data = Vec::<(Vec<f64>, Vec<f64>)>::new();
        for l in csv.lines().skip(1) {
            let row = l.split(',').skip(1).collect::<Vec<&str>>();
            let input = (0..4)
                .map(|i| row[i].parse::<f64>().unwrap())
                .collect::<Vec<f64>>();
            let mut label = vec![0.0; 3];
            label[match row[4] {
                "Iris-setosa" => 0,
                "Iris-versicolor" => 1,
                "Iris-virginica" => 2,
                _ => unreachable!(),
            }] = 1.0;
            data.push((input, label));
        }
        Self::new(data)
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
            100,
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
                        vec![((samples / 4..samples * 3 / 4).contains(&i)) as u8 as f64],
                    )
                })
                .collect(),
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

    /// returns 80% train, 20% test
    pub fn split(self) -> (Self, Self) {
        let mut a = self.0;
        let b = a.split_off(a.len() * 4 / 5);
        (Self::new(a), Self::new(b))
    }

    pub fn shuffle(&mut self) {
        self.0.shuffle(&mut rand::rng());
    }

    /// access to internal data via immutable slice
    pub fn data(&self) -> &[(Vec<f64>, Vec<f64>)] {
        &self.0
    }
}

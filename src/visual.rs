/*
use itertools::Itertools;
use macroquad::{
    color::{BLACK, BLUE, Color, GRAY, GREEN, RED, WHITE},
    shapes::{draw_circle, draw_line, draw_rectangle},
    text::draw_text,
    window::{screen_height, screen_width},
};

use crate::{algos::knn_classify, data::Dataset, linalg::Matrix, neural::NeuralNet};

#[derive(Debug, Clone, Copy)]
pub struct WindowPos {
    w: f32,
    h: f32,
    dw: f32,
    dh: f32,
}

impl WindowPos {
    pub fn new(w: f32, h: f32, dw: f32, dh: f32) -> Self {
        Self { w, h, dw, dh }
    }
    pub fn full() -> Self {
        Self::new(1.0, 1.0, 0.0, 0.0)
    }
    pub fn split_h() -> [Self; 2] {
        [Self::new(1.0, 0.5, 0.0, 0.0), Self::new(1.0, 0.5, 0.0, 0.5)]
    }
    pub fn split_v() -> [Self; 2] {
        [Self::new(0.5, 1.0, 0.0, 0.0), Self::new(0.5, 1.0, 0.5, 0.0)]
    }
    pub fn split_three() -> [Self; 3] {
        [
            Self::new(0.5, 1.0, 0.0, 0.0),
            Self::new(0.5, 0.5, 0.5, 0.0),
            Self::new(0.5, 0.5, 0.5, 0.5),
        ]
    }
    pub fn split_quad() -> [Self; 4] {
        [
            Self::new(0.5, 0.5, 0.0, 0.0),
            Self::new(0.5, 0.5, 0.0, 0.5),
            Self::new(0.5, 0.5, 0.5, 0.0),
            Self::new(0.5, 0.5, 0.5, 0.5),
        ]
    }
    pub fn map_x(self, x: f32) -> f32 {
        (x * self.w + self.dw) * screen_width()
    }
    pub fn map_y(self, y: f32) -> f32 {
        (y * self.h + self.dh) * screen_height()
    }
    pub fn clear(self, color: Color) {
        draw_rectangle(
            self.dw * screen_width(),
            self.dh * screen_height(),
            self.w * screen_width(),
            self.h * screen_height(),
            color,
        );
    }
}

pub enum Border {
    Discrete,
    Lerp,
    White,
    Black,
}

pub fn draw_accuracy(train: &[f64], test: &[f64], win: WindowPos) {
    assert_eq!(train.len(), test.len());
    let n = train.len();
    win.clear(WHITE);

    for i in 0..10 {
        draw_line(
            win.map_x(0.0),
            win.map_y(i as f32 / 10.0),
            win.map_x(1.0),
            win.map_y(i as f32 / 10.0),
            1.0,
            GRAY,
        );
    }
    for (i, &[a, b]) in train.array_windows().enumerate() {
        draw_line(
            win.map_x(i as f32 / (n - 1) as f32),
            win.map_y(1.0 - a as f32),
            win.map_x((i + 1) as f32 / (n - 1) as f32),
            win.map_y(1.0 - b as f32),
            3.0,
            RED,
        );
    }
    for (i, &[a, b]) in test.array_windows().enumerate() {
        draw_line(
            win.map_x(i as f32 / (n - 1) as f32),
            win.map_y(1.0 - a as f32),
            win.map_x((i + 1) as f32 / (n - 1) as f32),
            win.map_y(1.0 - b as f32),
            3.0,
            GREEN,
        );
    }
    draw_text(
        format!("train accuracy: {:.2}%", train[n - 1] * 100.0),
        win.map_x(0.0),
        win.map_y(0.75),
        30.0,
        RED,
    );
    draw_text(
        format!("test accuracy:  {:.2}%", test[n - 1] * 100.0),
        win.map_x(0.0),
        win.map_y(0.82),
        30.0,
        GREEN,
    );
}

pub fn draw_loss(train: &[f64], test: &[f64], win: WindowPos) {
    assert_eq!(train.len(), test.len());
    let n = train.len();
    win.clear(WHITE);

    let (min, max) = test
        .iter()
        .chain(train.iter())
        .fold((f64::MAX, f64::MIN), |(min, max), &x| {
            (min.min(x), max.max(x))
        });

    for (i, [a, b]) in train
        .iter()
        .map(|&x| (x - min) / (max - min))
        .array_windows()
        .enumerate()
    {
        draw_line(
            win.map_x(i as f32 / (n - 1) as f32),
            win.map_y(1.0 - a as f32),
            win.map_x((i + 1) as f32 / (n - 1) as f32),
            win.map_y(1.0 - b as f32),
            3.0,
            RED,
        );
    }
    for (i, [a, b]) in test
        .iter()
        .map(|&x| (x - min) / (max - min))
        .array_windows()
        .enumerate()
    {
        draw_line(
            win.map_x(i as f32 / (n - 1) as f32),
            win.map_y(1.0 - a as f32),
            win.map_x((i + 1) as f32 / (n - 1) as f32),
            win.map_y(1.0 - b as f32),
            3.0,
            GREEN,
        );
    }

    draw_text(
        format!("cross entropy loss:  {:.2}", test[n - 1] * 100.0),
        win.map_x(0.0),
        win.map_y(0.15),
        30.0,
        GRAY,
    );
}
pub fn draw_colored_plane_net(net: &NeuralNet, dataset: &Dataset, border: Border, win: WindowPos) {
    win.clear(BLACK);
    // 0 = red, 1 = blue, 2 = green
    let resolution = 100;
    for i in 0..resolution {
        for j in 0..resolution {
            let x = i as f32 / resolution as f32;
            let y = (j + 1) as f32 / resolution as f32;
            let w = 1.0 / resolution as f32;
            let h = 1.0 / resolution as f32;
            let o = net.forward(Matrix::new_row(vec![x as f64, y as f64]));
            let color = if net.dimensions().1 == 3 {
                let [r, b, g] = [0, 1, 2].map(|i| o.data[i] as f32);
                match border {
                    Border::Lerp => Color::new(r, g, b, 0.5),
                    Border::White | Border::Black => panic!(),
                    Border::Discrete => if r > b {
                        if r > g { RED } else { GREEN }
                    } else {
                        if b > g { BLUE } else { GREEN }
                    }
                    .with_alpha(0.5),
                }
            } else {
                let [r, b] = [0, 1].map(|i| o.data[i] as f32);
                match border {
                    Border::Lerp => Color::new(r, 0.0, b, 0.5),
                    Border::Discrete => if r > 0.5 { RED } else { BLUE }.with_alpha(0.5),
                    Border::White => {
                        if r > 0.5 {
                            // red -> white
                            Color::new(1.0, 1.0 - 2.0 * (r - 0.5), 1.0 - 2.0 * (r - 0.5), 0.5)
                        } else {
                            // blue -> white
                            Color::new(1.0 - 2.0 * (b - 0.5), 1.0 - 2.0 * (b - 0.5), 1.0, 0.5)
                        }
                    }
                    Border::Black => {
                        if r > 0.5 {
                            // red -> black
                            Color::new(2.0 * (r - 0.5), 0.0, 0.0, 0.5)
                        } else {
                            // blue -> black
                            Color::new(0.0, 0.0, 2.0 * (b - 0.5), 0.5)
                        }
                    }
                }
            };

            draw_rectangle(
                win.map_x(x),
                win.map_y(1.0 - y),
                w * screen_width() * win.w,
                h * screen_height() * win.h,
                color,
            );
        }
    }
    for (input, label) in dataset.data() {
        let (x, y) = (input[0], input[1]);
        draw_circle(
            win.map_x(x as f32),
            win.map_y(1.0 - y as f32),
            0.003 * screen_width(),
            match *label {
                0 => RED,
                1 => BLUE,
                2 => GREEN,
                _ => panic!(),
            },
        );
    }
}

pub fn draw_colored_plane_knn(dataset: &mut Dataset, k: usize, win: WindowPos) {
    win.clear(BLACK);
    let resolution = 100;
    for i in 0..resolution {
        for j in 0..resolution {
            let x = i as f32 / resolution as f32;
            let y = (j + 1) as f32 / resolution as f32;
            let w = 1.0 / resolution as f32;
            let h = 1.0 / resolution as f32;
            let label = knn_classify(dataset, &[x as f64, y as f64], k, false);
            let color = if dataset.dimensions().1 == 3 {
                match label {
                    0 => RED,
                    1 => BLUE,
                    2 => GREEN,
                    _ => panic!(),
                }
                .with_alpha(0.5)
            } else {
                if label == 0 { RED } else { BLUE }.with_alpha(0.5)
            };
            draw_rectangle(
                win.map_x(x),
                win.map_y(1.0 - y),
                w * screen_width() * win.w,
                h * screen_height() * win.h,
                color,
            );
        }
    }
    for (input, label) in dataset.data() {
        let (x, y) = (input[0], input[1]);
        draw_circle(
            win.map_x(x as f32),
            win.map_y(1.0 - y as f32),
            0.003 * screen_width(),
            match *label {
                0 => RED,
                1 => BLUE,
                2 => GREEN,
                _ => panic!(),
            },
        );
    }
}
*/

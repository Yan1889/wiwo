use itertools::Itertools;
use macroquad::{
    color::{BLACK, BLUE, Color, GRAY, GREEN, RED, WHITE},
    shapes::{draw_circle, draw_line, draw_rectangle},
    text::draw_text,
    window::{screen_height, screen_width},
};

use crate::{data::Dataset, linalg::Matrix, neural::NeuralNet};

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

pub fn draw_accuracy(train: &[f64], test: &[f64], win: WindowPos) {
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
            win.map_x(i as f32 / (train.len() - 1) as f32),
            win.map_y(1.0 - a as f32),
            win.map_x((i + 1) as f32 / (train.len() - 1) as f32),
            win.map_y(1.0 - b as f32),
            3.0,
            RED,
        );
    }
    for (i, &[a, b]) in test.array_windows().enumerate() {
        draw_line(
            win.map_x(i as f32 / (test.len() - 1) as f32),
            win.map_y(1.0 - a as f32),
            win.map_x((i + 1) as f32 / (test.len() - 1) as f32),
            win.map_y(1.0 - b as f32),
            3.0,
            GREEN,
        );
    }
    draw_text(
        format!("train accuracy: {:.2}%", train.last().unwrap() * 100.0),
        win.map_x(0.0),
        win.map_y(0.75),
        30.0,
        RED,
    );
    draw_text(
        format!("test accuracy:  {:.2}%", test.last().unwrap() * 100.0),
        win.map_x(0.0),
        win.map_y(0.82),
        30.0,
        GREEN,
    );
}

pub fn draw_loss(history: &[f64], win: WindowPos) {
    win.clear(WHITE);
    let min = history.iter().fold(f64::MAX, |acc, &x| acc.min(x));
    let max = history.iter().fold(f64::MIN, |acc, &x| acc.max(x));
    for (i, [a, b]) in history
        .iter()
        .map(|&x| (x - min) / (max - min))
        .array_windows()
        .enumerate()
    {
        draw_line(
            win.map_x(i as f32 / (history.len() - 1) as f32),
            win.map_y(1.0 - a as f32),
            win.map_x((i + 1) as f32 / (history.len() - 1) as f32),
            win.map_y(1.0 - b as f32),
            3.0,
            BLACK,
        );
    }

    draw_text(
        format!(
            "cross entropy loss:  {:.2}",
            history.last().unwrap() * 100.0
        ),
        win.map_x(0.0),
        win.map_y(0.15),
        30.0,
        GRAY,
    );
}
pub fn draw_colored_plane(net: &NeuralNet, dataset: &Dataset, discrete: bool, win: WindowPos) {
    win.clear(BLACK);
    let resolution = 100;
    for i in 0..resolution {
        for j in 0..resolution {
            let x = i as f32 / resolution as f32;
            let y = (j + 1) as f32 / resolution as f32;
            let w = 1.0 / resolution as f32;
            let h = 1.0 / resolution as f32;
            let r = net
                .forward(Matrix::new_column(vec![x as f64, y as f64]))
                .data()[1][0];

            draw_rectangle(
                win.map_x(x),
                win.map_y(1.0 - y),
                w * screen_width() * win.w,
                h * screen_height() * win.h,
                if discrete {
                    if r < 0.5 {
                        RED.with_alpha(0.5)
                    } else {
                        BLUE.with_alpha(0.5)
                    }
                } else {
                    // RED -> White -> Blue
                    // 1, 0, 0 -> 1, 1, 1 -> 0, 0, 1
                    // Color::new(1.0 - r as f32, 0.0, r as f32, 0.5)
                    if r < 0.5 {
                        Color::new(1.0, 2.0 * r as f32, 2.0 * r as f32, 0.5)
                    } else {
                        Color::new(
                            1.0 - 2.0 * (r - 0.5) as f32,
                            1.0 - 2.0 * (r - 0.5) as f32,
                            1.0,
                            0.5,
                        )
                    }
                },
            );
        }
    }
    for (input, label) in dataset.data() {
        let x = input[0];
        let y = input[1];
        draw_circle(
            win.map_x(x as f32),
            win.map_y(1.0 - y as f32),
            0.003 * screen_width(),
            if *label == 1 { BLUE } else { RED },
        );
    }
}

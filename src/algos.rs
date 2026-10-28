use std::collections::HashSet;

use crate::data::Dataset;

pub fn knn_classify(dataset: &Dataset, p: &[f64], k: usize, manhattan: bool) -> usize {
    let mut distances = Vec::new();
    for known in &dataset.data {
        let d = known
            .0
            .iter()
            .zip(p.iter())
            .map(|(x, y)| {
                if manhattan {
                    (x - y).abs()
                } else {
                    // euclid
                    (x - y).powi(2)
                }
            })
            .sum::<f64>();
        distances.push(if manhattan { d } else { d.sqrt() });
    }
    let mut votes = vec![0; dataset.classes];
    let mut used = HashSet::new();
    for _ in 0..k {
        let mut best = usize::MAX;
        for i in 0..dataset.data.len() {
            if used.contains(&i) {
                continue;
            }
            if best == usize::MAX || distances[i] < distances[best] {
                best = i;
            }
        }
        used.insert(best);
        votes[dataset.data[best].1] += 1;
    }
    votes
        .into_iter()
        .enumerate()
        .max_by_key(|(_, n)| *n)
        .unwrap()
        .0
}

pub fn knn_accuracy(train: &Dataset, test: &Dataset, k: usize, manhattan: bool) -> f64 {
    let mut count = 0;
    for (input, label) in &test.data {
        if knn_classify(train, input, k, manhattan) == *label {
            count += 1;
        }
    }
    count as f64 / test.data.len() as f64
}

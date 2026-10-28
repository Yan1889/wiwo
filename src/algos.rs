use std::collections::HashSet;

use crate::data::Dataset;

pub fn knn_classify(dataset: &Dataset, p: &[f64], k: usize) -> usize {
    let mut distances = Vec::new();
    for known in &dataset.data {
        // manhattan distance
        distances.push(
            known
                .0
                .iter()
                .zip(p.iter())
                .map(|(x, y)| (x - y).abs())
                .sum::<f64>(),
        );
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

pub fn knn_accuracy(train: &Dataset, test: &Dataset, k: usize) -> f64 {
    let mut count = 0;
    for (input, label) in &test.data {
        if knn_classify(train, input, k) == *label {
            count += 1;
        }
    }
    count as f64 / test.data.len() as f64
}

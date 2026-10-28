use crate::data::Dataset;

pub fn knn_classify(dataset: &mut Dataset, p: &[f64], k: usize) -> usize {
    dataset.knn_sort(p);
    // take first k elements
    let mut votes = vec![0; dataset.classes];
    for other in dataset.data.iter().take(k) {
        votes[other.1] += 1;
    }
    // take the one with the most votes
    votes
        .into_iter()
        .enumerate()
        .max_by_key(|(_, n)| *n)
        .unwrap()
        .0
}

pub fn knn_accuracy(train: &mut Dataset, test: &Dataset, k: usize) -> f64 {
    let mut count = 0;
    for (input, label) in &test.data {
        if knn_classify(train, input, k) == *label {
            count += 1;
        }
    }
    count as f64 / test.data.len() as f64
}

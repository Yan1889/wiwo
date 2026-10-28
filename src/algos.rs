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
        .1
}

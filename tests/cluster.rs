//! Union-find clustering of edge lists, including clustering the output of the
//! candidate finders via the edge `pair()` helper.

use std::collections::{HashMap, HashSet};

use perceptual_dedupe::align::find_candidates;
use perceptual_dedupe::cluster::cluster_edges;
use perceptual_dedupe::config::DedupParams;

fn sorted_groups(mut groups: Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    for g in &mut groups {
        g.sort_unstable();
    }
    groups.sort_unstable();
    groups
}

#[test]
fn three_disjoint_pairs() {
    let groups = sorted_groups(cluster_edges(&[(10u64, 11), (20, 21), (30, 31)]));
    assert_eq!(groups, vec![vec![10, 11], vec![20, 21], vec![30, 31]]);
}

#[test]
fn chain_collapses_to_one_component() {
    let groups = cluster_edges(&[(1u64, 2), (2, 3), (3, 4), (4, 5)]);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 5);
}

#[test]
fn duplicate_edges_do_not_double_count_members() {
    let groups = cluster_edges(&[(1u64, 2), (1, 2), (2, 1)]);
    assert_eq!(groups.len(), 1);
    let mut members = groups[0].clone();
    members.sort_unstable();
    assert_eq!(members, vec![1, 2]);
}

#[test]
fn clusters_candidate_edges_through_pair_helper() {
    // Three identical clips should land in one duplicate group once their
    // pairwise edges are clustered.
    let params = DedupParams::default();
    let seq: Vec<u64> = (0..40u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let mut hashes: HashMap<u64, Vec<u64>> = HashMap::new();
    for id in [1u64, 2, 3] {
        hashes.insert(id, seq.clone());
    }
    let ranks: HashMap<u64, u64> = [(1u64, 0), (2, 1), (3, 2)].into_iter().collect();
    let edges = find_candidates(&hashes, &ranks, &HashSet::new(), &params);
    let pairs: Vec<(u64, u64)> = edges.iter().map(|e| e.pair()).collect();
    let groups = cluster_edges(&pairs);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 3);
}

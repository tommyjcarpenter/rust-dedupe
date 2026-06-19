//! Union-find clustering: turn a list of duplicate edges into connected
//! components.
//!
//! The matching primitives produce pairwise edges ("a is a near-duplicate of
//! b"). Duplicate relationships are transitive in practice — if a matches b
//! and b matches c, the three belong in one set — so the final step is to take
//! the connected components of the edge graph.

use std::collections::HashMap;
use std::hash::Hash;

/* Iterative find with path-halving. Operates on interned usize indices so it
is independent of the caller's Id type. */
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Group ids into connected components by union-find over `edges`.
///
/// Every id that appears in any edge is placed in exactly one returned group.
/// Ids never seen in an edge are not in the output (the function only knows
/// about ids it was given). Groups, and the ids within them, are emitted in
/// first-appearance order, so the result is deterministic regardless of the
/// `Id` type's hashing.
///
/// To cluster the output of [`crate::align::find_candidates`] or
/// [`crate::audio::find_audio_candidates`], map each edge through its
/// `pair()` method first.
pub fn cluster_edges<Id: Copy + Eq + Hash>(edges: &[(Id, Id)]) -> Vec<Vec<Id>> {
    let mut index: HashMap<Id, usize> = HashMap::new();
    let mut ids: Vec<Id> = Vec::new();
    let mut parent: Vec<usize> = Vec::new();

    for &(a, b) in edges {
        for id in [a, b] {
            if let std::collections::hash_map::Entry::Vacant(slot) = index.entry(id) {
                let n = ids.len();
                slot.insert(n);
                ids.push(id);
                parent.push(n);
            }
        }
    }

    for &(a, b) in edges {
        let ra = find(&mut parent, index[&a]);
        let rb = find(&mut parent, index[&b]);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    let mut groups: Vec<Vec<Id>> = Vec::new();
    let mut root_to_group: HashMap<usize, usize> = HashMap::new();
    for (i, &id) in ids.iter().enumerate() {
        let root = find(&mut parent, i);
        let gi = *root_to_group.entry(root).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[gi].push(id);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separate_pairs_form_separate_groups() {
        let groups = cluster_edges(&[(1u32, 2), (3, 4)]);
        assert_eq!(groups, vec![vec![1, 2], vec![3, 4]]);
    }

    #[test]
    fn transitive_edges_merge() {
        // 1-2, 2-3 => one group; 4-5 separate.
        let groups = cluster_edges(&[(1u32, 2), (2, 3), (4, 5)]);
        assert_eq!(groups.len(), 2);
        let mut first = groups[0].clone();
        first.sort_unstable();
        assert_eq!(first, vec![1, 2, 3]);
    }

    #[test]
    fn empty_edges_yield_no_groups() {
        let groups: Vec<Vec<u32>> = cluster_edges(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn diamond_merges_into_one() {
        // 1-2, 1-3, 2-4, 3-4 => single component of 4.
        let groups = cluster_edges(&[(1u32, 2), (1, 3), (2, 4), (3, 4)]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 4);
    }
}

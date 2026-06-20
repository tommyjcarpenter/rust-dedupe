//! Property tests over the pure primitives.

use std::collections::HashSet;

use perceptual_dedupe::align::best_alignment;
use perceptual_dedupe::cluster::cluster_edges;
use perceptual_dedupe::frame_hash::hamming64;
use perceptual_dedupe::image_hash::{GRAY_LEN, ImageHash};
use proptest::prelude::*;

proptest! {
    #[test]
    fn hamming_is_symmetric(a in any::<u64>(), b in any::<u64>()) {
        prop_assert_eq!(hamming64(a, b), hamming64(b, a));
    }

    #[test]
    fn hamming_obeys_triangle_inequality(a in any::<u64>(), b in any::<u64>(), c in any::<u64>()) {
        prop_assert!(hamming64(a, c) <= hamming64(a, b) + hamming64(b, c));
    }

    #[test]
    fn hamming_with_self_is_zero(a in any::<u64>()) {
        prop_assert_eq!(hamming64(a, a), 0);
    }

    // With the motion gate disabled, a sequence aligned against itself has a
    // perfect zero-distance alignment at shift 0, so the minimum average is 0.
    #[test]
    fn self_alignment_scores_zero(seq in prop::collection::vec(any::<u64>(), 30..50)) {
        let al = best_alignment(&seq, &seq, 10, 0);
        prop_assert_eq!(al.avg_bits, 0.0);
        prop_assert!(al.overlap >= 10);
    }

    // The motion mask and the XOR both commute, so the set of achievable
    // alignments — and hence the minimum average — is identical in either
    // direction. (The reported `overlap` is not asserted: when several shifts
    // tie on the minimum average, the first-wins tie-break can pick different
    // overlaps depending on iteration direction.)
    #[test]
    fn alignment_average_is_order_independent(
        a in prop::collection::vec(any::<u64>(), 30..50),
        b in prop::collection::vec(any::<u64>(), 30..50),
    ) {
        let ab = best_alignment(&a, &b, 10, 2);
        let ba = best_alignment(&b, &a, 10, 2);
        prop_assert_eq!(ab.avg_bits, ba.avg_bits);
    }

    #[test]
    fn image_hash_hex_round_trips(gray in prop::collection::vec(any::<u8>(), GRAY_LEN..=GRAY_LEN)) {
        let h = ImageHash::from_gray_rows(&gray).unwrap();
        prop_assert_eq!(ImageHash::from_hex(&h.to_hex()).unwrap(), h);
    }

    // Clustering must yield a partition: every id seen in an edge appears in
    // exactly one group, and both endpoints of every edge share a group.
    #[test]
    fn clustering_is_a_valid_partition(edges in prop::collection::vec((0u8..12, 0u8..12), 0..40)) {
        let groups = cluster_edges(&edges);

        // Every id appears exactly once.
        let mut seen: HashSet<u8> = HashSet::new();
        for g in &groups {
            for &id in g {
                prop_assert!(seen.insert(id), "id {} appeared in two groups", id);
            }
        }

        // Coverage: exactly the ids that appear in some edge.
        let edge_ids: HashSet<u8> = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
        prop_assert_eq!(&seen, &edge_ids);

        // Connectivity: endpoints of each edge land in the same group.
        let group_of = |id: u8| groups.iter().position(|g| g.contains(&id));
        for &(a, b) in &edges {
            prop_assert_eq!(group_of(a), group_of(b));
        }
    }
}

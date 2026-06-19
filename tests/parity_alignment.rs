//! Sliding-alignment parity vectors from both source codebases.
//!
//! With `motion_bits = 0` the crate reproduces the plain
//! average-over-all-overlapping-frames behavior (the simpler port); with
//! `motion_bits = 2` it is the motion-gated scoring. Both sets of source
//! vectors are pinned here against the one function.

use std::collections::{HashMap, HashSet};

use perceptual_dedupe::align::{
    OverlapKind, best_alignment, find_candidates, score_visual, score_visual_segments,
};
use perceptual_dedupe::config::DedupParams;

// A sequence whose consecutive frames differ by many bits, so every frame
// clears the motion gate.
fn moving_seq(n: u64) -> Vec<u64> {
    (0..n)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect()
}

// ---------------------------------------------------------------------------
// Plain averaging (motion_bits = 0) — the simpler port's vectors.
// ---------------------------------------------------------------------------

#[test]
fn plain_perfect_match_at_zero_shift() {
    let seq = vec![0xAAu64, 0xBB, 0xCC, 0xDD, 0xEE];
    let al = best_alignment(&seq, &seq, 1, 0);
    assert_eq!((al.shift, al.avg_bits, al.overlap), (0, 0.0, 5));
}

#[test]
fn plain_positive_shift_trimmed_front() {
    let a = vec![0x11u64, 0x22, 0x33, 0x44, 0x55, 0x66];
    let b = vec![0x33u64, 0x44, 0x55, 0x66];
    let al = best_alignment(&a, &b, 1, 0);
    assert_eq!((al.shift, al.avg_bits, al.overlap), (2, 0.0, 4));
}

#[test]
fn plain_negative_shift_when_a_is_trimmed() {
    let a = vec![0x33u64, 0x44, 0x55, 0x66];
    let b = vec![0x11u64, 0x22, 0x33, 0x44, 0x55, 0x66];
    let al = best_alignment(&a, &b, 1, 0);
    assert_eq!((al.shift, al.avg_bits, al.overlap), (-2, 0.0, 4));
}

#[test]
fn plain_min_overlap_floor_suppresses_degenerate_match() {
    let a = vec![
        0x11u64, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA,
    ];
    let b = vec![0x44u64, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA];
    let al = best_alignment(&a, &b, 5, 0);
    assert_eq!((al.shift, al.avg_bits, al.overlap), (3, 0.0, 7));
}

#[test]
fn plain_no_alignment_meets_min_returns_sentinel() {
    let al = best_alignment(&[0x11u64, 0x22], &[0x33u64, 0x44], 100, 0);
    assert_eq!(al.avg_bits, f32::MAX);
    assert_eq!(al.overlap, 0);
}

#[test]
fn plain_short_b_inside_much_longer_a() {
    let a: Vec<u64> = (0..60).collect();
    let b = a[20..40].to_vec();
    let al = best_alignment(&a, &b, 8, 0);
    assert_eq!((al.shift, al.avg_bits, al.overlap), (20, 0.0, 20));
}

#[test]
fn plain_short_a_inside_much_longer_b() {
    let b: Vec<u64> = (0..60).collect();
    let a = b[20..40].to_vec();
    let al = best_alignment(&a, &b, 8, 0);
    assert_eq!((al.shift, al.avg_bits, al.overlap), (-20, 0.0, 20));
}

#[test]
fn plain_partial_diff_averages_correctly() {
    let a = vec![0u64; 8];
    let mut b = vec![0u64; 8];
    b[3] = 0x01;
    let al = best_alignment(&a, &b, 8, 0);
    assert_eq!(al.shift, 0);
    assert_eq!(al.overlap, 8);
    assert!((al.avg_bits - 1.0 / 8.0).abs() < 1e-6);
}

#[test]
fn empty_inputs_are_safe() {
    assert_eq!(best_alignment(&[], &[1u64], 1, 0).overlap, 0);
    assert_eq!(best_alignment(&[1u64], &[], 1, 0).overlap, 0);
    assert_eq!(best_alignment(&[], &[], 1, 0).overlap, 0);
}

// ---------------------------------------------------------------------------
// Motion-gated scoring (motion_bits = 2) — the source-of-truth vectors.
// ---------------------------------------------------------------------------

#[test]
fn gated_identical_moving_sequence_scores_zero() {
    let seq = moving_seq(40);
    let al = best_alignment(&seq, &seq, 8, 2);
    assert_eq!(al.shift, 0);
    assert_eq!(al.avg_bits, 0.0);
    assert_eq!(al.overlap, 40);
}

#[test]
fn gated_uniform_xor_mask_gives_popcount_average() {
    let mask = 0x0000_0000_0000_0FFFu64; // 12 bits
    let a = moving_seq(40);
    let b: Vec<u64> = a.iter().map(|&x| x ^ mask).collect();
    let al = best_alignment(&a, &b, 30, 2);
    assert_eq!(al.avg_bits, 12.0);
    assert!(al.overlap >= 30);
}

#[test]
fn gated_two_static_runs_do_not_match() {
    // A repeated frame (a watermark / end card) has no motion; two such runs
    // must not align at ~0 bits.
    let a = vec![0xABCDu64; 60];
    let b = vec![0xABCDu64; 60];
    assert!(!best_alignment(&a, &b, 30, 2).matched());
}

#[test]
fn gated_shared_static_card_is_rejected() {
    // Different moving content, same trailing frozen card. The only ~0-bit
    // alignment is card-vs-card; static frames are excluded, so it cannot win.
    let move_a: Vec<u64> = (0..40u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let move_b: Vec<u64> = (0..40u64)
        .map(|i| i.wrapping_mul(0xD6E8_FEB8_6659_FD93))
        .collect();
    let card = vec![0xC0FF_EE00u64; 40];
    let a: Vec<u64> = move_a.iter().chain(&card).copied().collect();
    let b: Vec<u64> = move_b.iter().chain(&card).copied().collect();
    assert!(best_alignment(&a, &b, 30, 2).avg_bits > 16.0);
}

#[test]
fn gated_score_independent_of_argument_order() {
    let mv = moving_seq(50);
    let card = vec![0xABCD_1234u64; 40];
    let with_card: Vec<u64> = mv.iter().chain(&card).copied().collect();
    let ab = best_alignment(&mv, &with_card, 30, 2);
    let ba = best_alignment(&with_card, &mv, 30, 2);
    assert_eq!(ab.avg_bits, ba.avg_bits);
    assert_eq!(ab.overlap, ba.overlap);
}

// ---------------------------------------------------------------------------
// score_visual / score_visual_segments (default params: gated, hard floor 30).
// ---------------------------------------------------------------------------

#[test]
fn score_visual_identical_is_zero() {
    let params = DedupParams::default();
    let seq = moving_seq(40);
    let score = score_visual(&seq, &seq, &params).expect("clears floor");
    assert_eq!(score.visual_avg, 0.0);
    assert_eq!(score.visual_overlap, 40);
    assert!(score.audio.is_none());
}

#[test]
fn score_visual_below_hard_floor_is_none() {
    let params = DedupParams::default();
    let short = moving_seq(params.min_overlap_hard_floor as u64 - 1);
    let long = moving_seq(params.min_overlap_hard_floor as u64 + 10);
    assert!(score_visual(&short, &long, &params).is_none());
    assert!(score_visual(&long, &short, &params).is_none());
}

#[test]
fn score_visual_static_run_is_none() {
    let params = DedupParams::default();
    let a = vec![0xABCDu64; 60];
    assert!(score_visual(&a, &a, &params).is_none());
}

#[test]
fn score_visual_segments_picks_best_segment_pair() {
    let params = DedupParams::default();
    let unrelated = moving_seq(40)
        .iter()
        .map(|&x| x ^ 0xDEAD_BEEF)
        .collect::<Vec<_>>();
    let shared = moving_seq(40);
    let a = vec![unrelated, shared.clone()];
    let b = vec![shared];
    let score = score_visual_segments(&a, &b, &params).expect("tail pair clears floor");
    assert_eq!(score.visual_avg, 0.0);
    assert_eq!(score.visual_seg_a, 1, "a's matched segment is its tail");
    assert_eq!(score.visual_seg_b, 0, "b's matched segment is its head");
}

// ---------------------------------------------------------------------------
// find_candidates — all-pairs over a map, canonical by rank.
// ---------------------------------------------------------------------------

fn rank(pairs: &[(u64, u64)]) -> HashMap<u64, u64> {
    pairs.iter().copied().collect()
}

#[test]
fn find_candidates_picks_lower_rank_as_canonical() {
    let params = DedupParams::default();
    let winner = 1u64;
    let loser = 2u64;
    let full = moving_seq(40);
    let trimmed = full[3..].to_vec();
    let mut hashes: HashMap<u64, Vec<u64>> = HashMap::new();
    hashes.insert(winner, full);
    hashes.insert(loser, trimmed);
    let ranks = rank(&[(winner, 0), (loser, 5)]);
    let edges = find_candidates(&hashes, &ranks, &HashSet::new(), &params);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].canonical, winner);
    assert_eq!(edges[0].member, loser);
    assert_eq!(edges[0].avg_bits, 0.0);
}

#[test]
fn find_candidates_ignores_below_overlap_minimum() {
    let params = DedupParams::default();
    let mut hashes: HashMap<u64, Vec<u64>> = HashMap::new();
    let same = moving_seq(3);
    hashes.insert(1, same.clone());
    hashes.insert(2, same);
    let ranks = rank(&[(1, 0), (2, 1)]);
    assert!(find_candidates(&hashes, &ranks, &HashSet::new(), &params).is_empty());
}

#[test]
fn find_candidates_rejects_above_threshold() {
    let params = DedupParams::default();
    // Moving content, but every frame differs by all 64 bits -> avg 64 > 10.
    let a = moving_seq(40);
    let b: Vec<u64> = a.iter().map(|&x| x ^ u64::MAX).collect();
    let mut hashes: HashMap<u64, Vec<u64>> = HashMap::new();
    hashes.insert(1, a);
    hashes.insert(2, b);
    let ranks = rank(&[(1, 0), (2, 1)]);
    assert!(find_candidates(&hashes, &ranks, &HashSet::new(), &params).is_empty());
}

#[test]
fn find_candidates_skips_user_excluded_pairs() {
    let params = DedupParams::default();
    let full = moving_seq(40);
    let mut hashes: HashMap<u64, Vec<u64>> = HashMap::new();
    hashes.insert(1, full.clone());
    hashes.insert(2, full);
    let ranks = rank(&[(1, 0), (2, 5)]);
    let mut excluded = HashSet::new();
    excluded.insert((1u64, 2u64)); // stored lower-first
    assert!(find_candidates(&hashes, &ranks, &excluded, &params).is_empty());
}

#[test]
fn classify_reports_containment() {
    let a = moving_seq(60);
    let b = a[20..40].to_vec();
    let al = best_alignment(&a, &b, 8, 2);
    assert_eq!(al.classify(a.len(), b.len()), OverlapKind::Contains);
}

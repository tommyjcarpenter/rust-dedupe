//! Audio sub-fingerprint alignment + candidate-aggregation parity vectors.

use std::collections::{HashMap, HashSet};

use perceptual_dedupe::audio::{
    best_audio_alignment, find_audio_candidates, hamming32, score_audio,
};
use perceptual_dedupe::config::DedupParams;

// ---------------------------------------------------------------------------
// hamming32
// ---------------------------------------------------------------------------

#[test]
fn hamming32_basics() {
    assert_eq!(hamming32(0, 0), 0);
    assert_eq!(hamming32(0, u32::MAX), 32);
    assert_eq!(hamming32(0, 1), 1);
}

// ---------------------------------------------------------------------------
// best_audio_alignment
// ---------------------------------------------------------------------------

#[test]
fn audio_perfect_match_at_zero_shift() {
    let seq: Vec<u32> = (0..60).collect();
    let al = best_audio_alignment(&seq, &seq, 50);
    assert_eq!((al.shift, al.avg_bits, al.overlap), (0, 0.0, 60));
}

#[test]
fn audio_positive_shift_b_is_tail() {
    let a: Vec<u32> = (0..60).collect();
    let b = a[5..].to_vec();
    let al = best_audio_alignment(&a, &b, 50);
    assert_eq!((al.shift, al.avg_bits, al.overlap), (5, 0.0, 55));
}

#[test]
fn audio_negative_shift_b_has_leading_extra() {
    let a: Vec<u32> = (0..60).collect();
    let mut b = vec![99u32; 5];
    b.extend(a.iter().copied());
    let al = best_audio_alignment(&a, &b, 50);
    assert_eq!((al.shift, al.avg_bits, al.overlap), (-5, 0.0, 60));
}

#[test]
fn audio_below_min_overlap_returns_sentinel() {
    let al = best_audio_alignment(&[1u32, 2, 3], &[1u32, 2, 3], 50);
    assert_eq!(al.avg_bits, f32::MAX);
    assert_eq!(al.overlap, 0);
}

#[test]
fn audio_one_bit_per_position_averages_one() {
    let a = vec![0u32; 50];
    let b = vec![1u32; 50];
    let al = best_audio_alignment(&a, &b, 50);
    assert_eq!(al.overlap, 50);
    assert_eq!(al.avg_bits, 1.0);
}

#[test]
fn audio_empty_inputs_are_safe() {
    assert_eq!(best_audio_alignment(&[], &[1u32], 1).overlap, 0);
    assert_eq!(best_audio_alignment(&[1u32], &[], 1).overlap, 0);
    assert_eq!(best_audio_alignment(&[], &[], 1).overlap, 0);
}

// ---------------------------------------------------------------------------
// score_audio
// ---------------------------------------------------------------------------

#[test]
fn score_audio_identical_is_zero() {
    let params = DedupParams::default();
    let fps: Vec<u32> = (0..40).collect();
    let (avg, overlap) = score_audio(&fps, &fps, &params).expect("clears floor");
    assert_eq!(avg, 0.0);
    assert_eq!(overlap, 40);
}

#[test]
fn score_audio_below_floor_is_none() {
    let params = DedupParams::default();
    let short = vec![0u32; params.audio_min_overlap_hard_floor - 1];
    let long = vec![0u32; params.audio_min_overlap_hard_floor + 10];
    assert!(score_audio(&short, &long, &params).is_none());
    assert!(score_audio(&long, &short, &params).is_none());
}

// ---------------------------------------------------------------------------
// find_audio_candidates
// ---------------------------------------------------------------------------

fn rank(pairs: &[(u64, u64)]) -> HashMap<u64, u64> {
    pairs.iter().copied().collect()
}

#[test]
fn audio_candidates_pick_lower_rank_as_canonical() {
    let params = DedupParams {
        audio_threshold_bits: 8.0,
        audio_min_overlap: 10,
        ..Default::default()
    };
    let winner = 1u64;
    let loser = 2u64;
    let seq: Vec<u32> = (0..50).map(|i| 0x0101_0101u32.wrapping_mul(i)).collect();
    let mut fps: HashMap<u64, Vec<u32>> = HashMap::new();
    fps.insert(winner, seq.clone());
    fps.insert(loser, seq);
    let ranks = rank(&[(winner, 0), (loser, 5)]);
    let edges = find_audio_candidates(&fps, &ranks, &HashSet::new(), &params);
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].canonical, winner);
    assert_eq!(edges[0].member, loser);
    assert_eq!(edges[0].avg_bits, 0.0);
    assert_eq!(edges[0].overlap_frames, 50);
}

#[test]
fn audio_candidates_reject_above_threshold() {
    let params = DedupParams {
        audio_threshold_bits: 8.0,
        ..Default::default()
    };
    let mut fps: HashMap<u64, Vec<u32>> = HashMap::new();
    fps.insert(1, vec![0u32; 50]);
    fps.insert(2, vec![u32::MAX; 50]);
    let ranks = rank(&[(1, 0), (2, 1)]);
    assert!(find_audio_candidates(&fps, &ranks, &HashSet::new(), &params).is_empty());
}

#[test]
fn audio_candidates_reject_below_hard_floor() {
    // Even with a permissive threshold and tiny target overlap, clips below
    // the hard floor cannot match.
    let params = DedupParams {
        audio_threshold_bits: 0.5,
        audio_min_overlap: 1,
        ..Default::default()
    };
    let mut fps: HashMap<u64, Vec<u32>> = HashMap::new();
    fps.insert(1, vec![0xAAu32; 5]);
    fps.insert(2, vec![0xAAu32; 5]);
    let ranks = rank(&[(1, 0), (2, 1)]);
    assert!(find_audio_candidates(&fps, &ranks, &HashSet::new(), &params).is_empty());
}

#[test]
fn audio_candidates_adaptive_floor_for_short_clips() {
    // Both clear the hard floor (40, 200) but the target (50) exceeds the
    // shorter clip; the adaptive floor lets them still match.
    let params = DedupParams {
        audio_threshold_bits: 8.0,
        audio_min_overlap: 50,
        ..Default::default()
    };
    let seq: Vec<u32> = (0..200).map(|i| 0x0101_0101u32.wrapping_mul(i)).collect();
    let short = seq[..40].to_vec();
    let mut fps: HashMap<u64, Vec<u32>> = HashMap::new();
    fps.insert(1, short);
    fps.insert(2, seq);
    let ranks = rank(&[(1, 0), (2, 1)]);
    assert_eq!(
        find_audio_candidates(&fps, &ranks, &HashSet::new(), &params).len(),
        1
    );
}

#[test]
fn audio_candidates_skip_excluded_pairs() {
    let params = DedupParams {
        audio_threshold_bits: 8.0,
        audio_min_overlap: 10,
        ..Default::default()
    };
    let seq: Vec<u32> = (0..50).map(|i| 0x0101_0101u32.wrapping_mul(i)).collect();
    let mut fps: HashMap<u64, Vec<u32>> = HashMap::new();
    fps.insert(1, seq.clone());
    fps.insert(2, seq);
    let ranks = rank(&[(1, 0), (2, 1)]);
    let mut excluded = HashSet::new();
    excluded.insert((1u64, 2u64));
    assert!(find_audio_candidates(&fps, &ranks, &excluded, &params).is_empty());
}

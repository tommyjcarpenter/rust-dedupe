//! Audio sub-fingerprint matching — a second, independent perceptual signal.
//!
//! A difference hash collapses to noise when a clip is re-encoded with
//! non-trivial transforms (color grading, crop, letterboxing): the pixels
//! change even though the content is the same. An acoustic fingerprint
//! (Chromaprint-style 32-bit sub-fingerprints) survives those, because audio
//! is locked to the video frame and re-encoding barely touches it. So an audio
//! match corroborates a borderline visual match.
//!
//! The alignment is the same sliding minimum-average-Hamming idea as
//! [`crate::align::best_alignment`], operating on 32-bit values, with the same
//! optional motion gate ([`DedupParams::audio_motion_bits`], off by default):
//! digital silence fingerprints to a constant run, which would otherwise align
//! against any other silent stretch at ~0 bits.
//!
//! Edges here are corroboration by default: a caller treats a strong audio
//! match as confirming at least weak visual similarity. The one exception is
//! [`DedupParams::audio_alone_bits`], a much tighter bar at which audio admits
//! a pair on its own — for content whose reframing puts it beyond what a coarse
//! frame hash can see at all.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::align::{Alignment, orient};
use crate::config::DedupParams;

/// Sample rate fed to the fingerprinter, in Hz. Fixed: it defines the audio
/// bytes the fingerprint is computed from, so changing it would make stored
/// fingerprints incomparable.
pub const AUDIO_SAMPLE_RATE: u32 = 22050;

/// Channel count fed to the fingerprinter (mono). Fixed, for the same reason
/// as [`AUDIO_SAMPLE_RATE`].
pub const AUDIO_CHANNELS: u32 = 1;

/// Hamming distance between two 32-bit sub-fingerprints.
#[inline]
pub fn hamming32(a: u32, b: u32) -> u32 {
    (a ^ b).count_ones()
}

/// Find the best alignment of `b` against `a` by sliding `b`'s sub-fingerprint
/// sequence and minimizing the average Hamming distance over the overlap.
///
/// Only alignments with overlap at least `min_overlap` are considered. Returns
/// [`Alignment::NO_MATCH`] when none qualifies.
///
/// `motion_bits` is the audio analogue of the visual motion gate: only
/// sub-fingerprints where either side differs from a neighbor by at least that
/// many bits are scored and counted toward the overlap, so a constant run —
/// digital silence, a held tone — cannot carry an alignment on its own. `0`
/// disables it, scoring the plain average over every overlapping
/// sub-fingerprint.
pub fn best_audio_alignment(
    a: &[u32],
    b: &[u32],
    min_overlap: usize,
    motion_bits: u32,
) -> Alignment {
    // At least one overlapping sub-fingerprint; treat 0 as 1 so the average is
    // never computed over an empty (0/0 -> NaN) overlap.
    let min_overlap = min_overlap.max(1);
    if a.is_empty() || b.is_empty() {
        return Alignment::NO_MATCH;
    }
    // See `align::best_alignment`: guard the i32 shift casts against
    // pathologically long inputs rather than truncating into bogus bounds.
    if a.len() > i32::MAX as usize || b.len() > i32::MAX as usize || min_overlap > i32::MAX as usize
    {
        return Alignment::NO_MATCH;
    }
    let max_pos: i32 = a.len() as i32 - min_overlap as i32;
    let max_neg: i32 = min_overlap as i32 - b.len() as i32;
    if max_pos < max_neg {
        return Alignment::NO_MATCH;
    }
    let moving_a = crate::align::moving_mask(a, motion_bits, hamming32_dist);
    let moving_b = crate::align::moving_mask(b, motion_bits, hamming32_dist);
    let mut best = Alignment::NO_MATCH;
    for shift in max_neg..=max_pos {
        let (a_start, b_start) = if shift >= 0 {
            (shift as usize, 0)
        } else {
            (0, (-shift) as usize)
        };
        let overlap = (a.len() - a_start).min(b.len() - b_start);
        if overlap < min_overlap {
            continue;
        }
        let mut moving_overlap = 0usize;
        // u64 accumulator: a long overlap can sum more than u32::MAX bits
        // (overlap up to i32::MAX, times 32 bits per sub-fingerprint).
        let mut total_bits = 0u64;
        let a_win = &a[a_start..a_start + overlap];
        let b_win = &b[b_start..b_start + overlap];
        let ma = &moving_a[a_start..a_start + overlap];
        let mb = &moving_b[b_start..b_start + overlap];
        for (((&ah, &bh), &am), &bm) in a_win.iter().zip(b_win).zip(ma).zip(mb) {
            if am || bm {
                moving_overlap += 1;
                total_bits += u64::from(hamming32(ah, bh));
            }
        }
        if moving_overlap < min_overlap {
            continue;
        }
        let avg = (total_bits as f64 / moving_overlap as f64) as f32;
        if avg < best.avg_bits {
            best = Alignment {
                shift,
                avg_bits: avg,
                overlap: moving_overlap,
            };
        }
    }
    best
}

/// [`hamming32`] over references, for [`crate::align::moving_mask`].
fn hamming32_dist(a: &u32, b: &u32) -> u32 {
    hamming32(*a, *b)
}

/// Score a single pair of sub-fingerprint sequences. `(avg_bits, overlap)`, or
/// `None` when either side is below the audio hard floor or no shift meets the
/// adaptive floor. Mirrors [`crate::align::score_visual`]'s floor logic on the
/// audio parameters.
pub fn score_audio(a: &[u32], b: &[u32], params: &DedupParams) -> Option<(f32, usize)> {
    if a.len() < params.audio_min_overlap_hard_floor
        || b.len() < params.audio_min_overlap_hard_floor
    {
        return None;
    }
    let effective_min = params
        .audio_min_overlap
        .max(params.audio_min_overlap_hard_floor)
        .min(a.len())
        .min(b.len());
    let alignment = best_audio_alignment(a, b, effective_min, params.audio_motion_bits);
    if alignment.overlap < effective_min {
        return None;
    }
    Some((alignment.avg_bits, alignment.overlap))
}

/// A candidate audio-corroborated duplicate edge, oriented so `canonical` is
/// the kept item. The audio analog of [`crate::align::DupeEdge`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioDupeEdge<Id> {
    /// The duplicate item.
    pub member: Id,
    /// The kept (lower-ranked) item.
    pub canonical: Id,
    /// Average Hamming distance per sub-fingerprint of the winning alignment.
    pub avg_bits: f32,
    /// Number of sub-fingerprints that average was taken over.
    pub overlap_frames: usize,
}

impl<Id: Copy> AudioDupeEdge<Id> {
    /// The `(canonical, member)` id pair, for feeding to
    /// [`crate::cluster::cluster_edges`].
    pub fn pair(&self) -> (Id, Id) {
        (self.canonical, self.member)
    }
}

/// All candidate audio edges over an all-pairs comparison of `fingerprints`.
/// The audio analog of [`crate::align::find_candidates`]: same rank /
/// exclusion / adaptive-floor rules, on the audio parameters. Edges here are
/// corroboration, not standalone verdicts.
pub fn find_audio_candidates<Id, V>(
    fingerprints: &HashMap<Id, V>,
    ranks: &HashMap<Id, u64>,
    excluded: &HashSet<(Id, Id)>,
    params: &DedupParams,
) -> Vec<AudioDupeEdge<Id>>
where
    Id: Copy + Eq + Hash + Ord,
    V: AsRef<[u32]>,
{
    // Sort by id so the edge order (and any downstream clustering order) is
    // reproducible rather than dependent on HashMap iteration order.
    let mut entries: Vec<(Id, &[u32])> =
        fingerprints.iter().map(|(k, v)| (*k, v.as_ref())).collect();
    entries.sort_unstable_by_key(|(id, _)| *id);
    let mut out = Vec::new();
    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            let (id_a, fa) = entries[i];
            let (id_b, fb) = entries[j];
            let pair = if id_a < id_b {
                (id_a, id_b)
            } else {
                (id_b, id_a)
            };
            if excluded.contains(&pair) {
                continue;
            }
            if fa.len() < params.audio_min_overlap_hard_floor
                || fb.len() < params.audio_min_overlap_hard_floor
            {
                continue;
            }
            let effective_min = params
                .audio_min_overlap
                .max(params.audio_min_overlap_hard_floor)
                .min(fa.len())
                .min(fb.len());
            let alignment = best_audio_alignment(fa, fb, effective_min, params.audio_motion_bits);
            if alignment.overlap < effective_min || alignment.avg_bits > params.audio_threshold_bits
            {
                continue;
            }
            let (canonical, member) = orient(id_a, id_b, ranks);
            out.push(AudioDupeEdge {
                member,
                canonical,
                avg_bits: alignment.avg_bits,
                overlap_frames: alignment.overlap,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_match_at_zero_shift() {
        let seq: Vec<u32> = (0..60).collect();
        let al = best_audio_alignment(&seq, &seq, 50, 0);
        assert_eq!(al.shift, 0);
        assert_eq!(al.avg_bits, 0.0);
        assert_eq!(al.overlap, 60);
    }

    #[test]
    fn one_bit_per_position_averages_one() {
        let a = vec![0u32; 50];
        let b = vec![1u32; 50];
        let al = best_audio_alignment(&a, &b, 50, 0);
        assert_eq!(al.overlap, 50);
        assert_eq!(al.avg_bits, 1.0);
    }

    #[test]
    fn zero_min_overlap_is_coerced_and_finite() {
        // min_overlap = 0 must not produce a 0/0 NaN; it is treated as 1.
        let seq: Vec<u32> = (0..40).collect();
        let al = best_audio_alignment(&seq, &seq, 0, 0);
        assert!(al.avg_bits.is_finite());
        assert_eq!(al.avg_bits, 0.0);
        assert!(al.overlap >= 1);
    }

    /* A sub-fingerprint sequence whose every consecutive pair differs by many bits, i.e. real
    audio as far as the motion gate is concerned. */
    fn moving_fp(n: u32) -> Vec<u32> {
        (0..n).map(|i| i.wrapping_mul(0x9E37_79B9)).collect()
    }

    #[test]
    fn the_gate_refuses_a_pair_that_shares_only_silence() {
        /* Digital silence fingerprints to a constant run. Ungated, two such stretches align at 0
        bits and are the strongest possible match, which would make every silent clip a duplicate
        of every other one. Gated, nothing is scored and there is no alignment at all. */
        let silence = [0u32; 60];
        let ungated = best_audio_alignment(&silence, &silence, 50, 0);
        assert_eq!(ungated.avg_bits, 0.0, "ungated, silence is a perfect match");
        assert!(ungated.matched());

        let gated = best_audio_alignment(&silence, &silence, 50, 2);
        assert_eq!(gated, Alignment::NO_MATCH);
    }

    #[test]
    fn the_gate_leaves_real_audio_alone() {
        // Every frame moves, so gating changes neither the average nor the overlap.
        let seq = moving_fp(60);
        assert_eq!(
            best_audio_alignment(&seq, &seq, 50, 2),
            best_audio_alignment(&seq, &seq, 50, 0)
        );
    }

    #[test]
    fn the_gate_excludes_a_trailing_silent_run_from_the_average() {
        /* Real audio followed by silence. The silent tail matches at 0 bits and would otherwise
        drag the average down, flattering a pair whose actual content differs. Gated, only the
        moving half is scored, so the average reports the content and the overlap counts it. */
        let quiet = [0u32; 40];
        let a: Vec<u32> = moving_fp(40)
            .into_iter()
            .chain(quiet.iter().copied())
            .collect();
        let mut b = a.clone();
        // Flip one bit in every moving frame: 1 bit/frame over the content, 0 over the silence.
        for x in b.iter_mut().take(40) {
            *x ^= 1;
        }
        let ungated = best_audio_alignment(&a, &b, 30, 0);
        assert_eq!(ungated.overlap, 80);
        assert_eq!(ungated.avg_bits, 0.5, "40 bits spread over all 80 frames");

        /* 41, not 40: `moving_mask` counts an element that differs from EITHER neighbour, so the
        first silent frame is counted on the strength of the moving frame before it. That boundary
        rule is deliberate — it stops an off-by-one dropping a minimum-length clip below the
        overlap floor — and the one extra frame contributes 0 bits, so it only dilutes by 1/41. */
        let gated = best_audio_alignment(&a, &b, 30, 2);
        assert_eq!(gated.overlap, 41, "the moving half plus one boundary frame");
        assert_eq!(
            gated.avg_bits,
            (40.0f64 / 41.0) as f32,
            "40 bits over the 41 frames that scored"
        );
        assert!(
            gated.avg_bits > ungated.avg_bits,
            "excluding the silence stops it flattering the pair"
        );
    }
}

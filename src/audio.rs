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
//! [`crate::align::best_alignment`], operating on 32-bit values, with no motion
//! gate. Edges here are not meant to stand alone: a caller treats a strong
//! audio match as corroboration of at least weak visual similarity, never as a
//! duplicate verdict on its own.

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
/// [`Alignment::NO_MATCH`] when none qualifies. Unlike the visual path there is
/// no motion gate; every overlapping sub-fingerprint is scored.
pub fn best_audio_alignment(a: &[u32], b: &[u32], min_overlap: usize) -> Alignment {
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
        // u64 accumulator: a long overlap can sum more than u32::MAX bits
        // (overlap up to i32::MAX, times 32 bits per sub-fingerprint).
        let total_bits: u64 = a[a_start..a_start + overlap]
            .iter()
            .zip(&b[b_start..b_start + overlap])
            .map(|(&x, &y)| u64::from((x ^ y).count_ones()))
            .sum();
        let avg = (total_bits as f64 / overlap as f64) as f32;
        if avg < best.avg_bits {
            best = Alignment {
                shift,
                avg_bits: avg,
                overlap,
            };
        }
    }
    best
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
    let alignment = best_audio_alignment(a, b, effective_min);
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
            let alignment = best_audio_alignment(fa, fb, effective_min);
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
        let al = best_audio_alignment(&seq, &seq, 50);
        assert_eq!(al.shift, 0);
        assert_eq!(al.avg_bits, 0.0);
        assert_eq!(al.overlap, 60);
    }

    #[test]
    fn one_bit_per_position_averages_one() {
        let a = vec![0u32; 50];
        let b = vec![1u32; 50];
        let al = best_audio_alignment(&a, &b, 50);
        assert_eq!(al.overlap, 50);
        assert_eq!(al.avg_bits, 1.0);
    }

    #[test]
    fn zero_min_overlap_is_coerced_and_finite() {
        // min_overlap = 0 must not produce a 0/0 NaN; it is treated as 1.
        let seq: Vec<u32> = (0..40).collect();
        let al = best_audio_alignment(&seq, &seq, 0);
        assert!(al.avg_bits.is_finite());
        assert_eq!(al.avg_bits, 0.0);
        assert!(al.overlap >= 1);
    }
}

//! Sliding-window alignment of frame-hash sequences, with motion-gated scoring.
//!
//! Two clips are compared by sliding one hash sequence against the other over
//! every integer shift and keeping the alignment that minimizes the average
//! Hamming distance per overlapping frame. Averaging (not summing) is
//! deliberate: a longer overlap is stronger evidence of shared content, so it
//! should not be penalized.
//!
//! # The motion gate
//!
//! A static run — a trailing watermark, a black or end card: the same frame
//! repeated — has near-zero frame-to-frame change, so two clips that share
//! only such a run align at ~0 bits and would falsely match. The motion gate
//! scores only frames where EITHER side is moving (differs from a neighbor by
//! at least `motion_bits`). The "either side" rule keeps the score symmetric,
//! so a verdict never depends on argument order.
//!
//! Setting `motion_bits = 0` disables the gate: every frame with a neighbor
//! counts, which reproduces the plain average-over-all-overlapping-frames
//! behavior. The two regimes are otherwise the same code path.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::config::DedupParams;

/// The result of aligning two hash sequences: the winning shift, the average
/// Hamming distance per scored (moving) frame, and how many frames that
/// average was taken over.
///
/// A positive `shift` means the second sequence starts later in the first
/// sequence's frame of reference (the first is trimmed at the front); a
/// negative shift means it starts earlier.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Alignment {
    /// Offset of `b` relative to `a` at the winning alignment.
    pub shift: i32,
    /// Average Hamming distance per scored frame. [`f32::MAX`] when no
    /// alignment met the overlap floor.
    pub avg_bits: f32,
    /// Number of (moving) frames the average was taken over. `0` when no
    /// alignment qualified.
    pub overlap: usize,
}

impl Alignment {
    /// The sentinel returned when no alignment meets the overlap floor.
    pub const NO_MATCH: Alignment = Alignment {
        shift: 0,
        avg_bits: f32::MAX,
        overlap: 0,
    };

    /// Whether a qualifying alignment was found.
    pub fn matched(&self) -> bool {
        self.overlap > 0
    }

    /// Classify how the two sequences overlap at this alignment, given their
    /// lengths. Lets a caller distinguish a strict superset/subset (one clip
    /// wholly contains the other) from a partial overlap. See [`OverlapKind`].
    ///
    /// Containment is judged by the geometric window the winning `shift`
    /// implies, not by [`Alignment::overlap`] — which counts only the frames
    /// the motion gate actually scored and is therefore smaller than the
    /// window whenever a static run was excluded.
    pub fn classify(&self, len_a: usize, len_b: usize) -> OverlapKind {
        if self.overlap == 0 {
            return OverlapKind::None;
        }
        let (a_start, b_start) = if self.shift >= 0 {
            (self.shift as usize, 0)
        } else {
            (0, (-self.shift) as usize)
        };
        let window = len_a
            .saturating_sub(a_start)
            .min(len_b.saturating_sub(b_start));
        let a_covered = window == len_a;
        let b_covered = window == len_b;
        match (a_covered, b_covered) {
            (true, true) => OverlapKind::Identical,
            (false, true) => OverlapKind::Contains,
            (true, false) => OverlapKind::ContainedBy,
            (false, false) => OverlapKind::Partial,
        }
    }
}

/// How two sequences sit relative to each other at a given alignment.
///
/// "a" and "b" are the first and second arguments passed to the alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OverlapKind {
    /// Both sequences are fully covered by the overlap (same content end to
    /// end).
    Identical,
    /// `a` wholly contains `b` (b is a strict subset of a).
    Contains,
    /// `b` wholly contains `a` (a is a strict subset of b).
    ContainedBy,
    /// The sequences overlap only partially; neither is wholly contained.
    Partial,
    /// No qualifying overlap.
    None,
}

/* Build a per-frame "moving" mask: frame k is moving if it differs from EITHER
neighbor by at least `motion_bits`. Using "either neighbor" keeps the boundary
frame of a moving run counted (no off-by-one that would drop a minimum-length
clip below the floor), while the interior of a frozen run counts zero.

`motion_bits == 0` disables the gate entirely: every frame is scored, including
a single-frame sequence (which has no neighbor to compare against). */
fn moving_mask(seq: &[u64], motion_bits: u32) -> Vec<bool> {
    if motion_bits == 0 {
        return vec![true; seq.len()];
    }
    (0..seq.len())
        .map(|k| {
            let prev = k > 0 && (seq[k] ^ seq[k - 1]).count_ones() >= motion_bits;
            let next = k + 1 < seq.len() && (seq[k] ^ seq[k + 1]).count_ones() >= motion_bits;
            prev || next
        })
        .collect()
}

/// Find the best alignment of `b` against `a` by sliding `b`'s hash sequence
/// and minimizing the average Hamming distance per scored frame.
///
/// Only frames where either side is moving (per `motion_bits`) are scored and
/// counted toward the overlap. Only alignments whose moving overlap is at
/// least `min_overlap` are considered. If none qualifies, returns
/// [`Alignment::NO_MATCH`].
///
/// The minimum `avg_bits` is order-independent: `best_alignment(a, b, ..)` and
/// `best_alignment(b, a, ..)` return the same `avg_bits`, because the moving
/// mask and the XOR both commute. The reported `shift` and `overlap` are not
/// guaranteed to mirror, though — when several shifts tie on the minimum
/// average, the first-wins tie-break can pick different ones in each direction.
pub fn best_alignment(a: &[u64], b: &[u64], min_overlap: usize, motion_bits: u32) -> Alignment {
    // An alignment needs at least one overlapping frame; treat 0 as 1 so the
    // average is never computed over an empty (0/0 -> NaN) overlap.
    let min_overlap = min_overlap.max(1);
    if a.is_empty() || b.is_empty() {
        return Alignment::NO_MATCH;
    }
    // The shift arithmetic below casts lengths to i32; sequences or an overlap
    // floor beyond i32::MAX would truncate into bogus bounds (and bad slice
    // indices). Such inputs are pathological (billions of frames), so treat
    // them as no match rather than misbehaving.
    if a.len() > i32::MAX as usize || b.len() > i32::MAX as usize || min_overlap > i32::MAX as usize
    {
        return Alignment::NO_MATCH;
    }
    /* Valid shift range, derived from the overlap constraint:
       positive shift: a_start = shift   -> overlap = min(a.len()-shift, b.len())
       negative shift: b_start = -shift   -> overlap = min(a.len(), b.len()+shift)
    The floor here is on raw overlap; the stricter moving-frame floor is applied
    per shift below. */
    let max_pos: i32 = a.len() as i32 - min_overlap as i32;
    let max_neg: i32 = min_overlap as i32 - b.len() as i32;
    if max_pos < max_neg {
        return Alignment::NO_MATCH;
    }
    let moving_a = moving_mask(a, motion_bits);
    let moving_b = moving_mask(b, motion_bits);
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
        let mut total_bits = 0u32;
        let a_win = &a[a_start..a_start + overlap];
        let b_win = &b[b_start..b_start + overlap];
        let ma = &moving_a[a_start..a_start + overlap];
        let mb = &moving_b[b_start..b_start + overlap];
        for (((&ah, &bh), &am), &bm) in a_win.iter().zip(b_win).zip(ma).zip(mb) {
            if am || bm {
                moving_overlap += 1;
                total_bits += (ah ^ bh).count_ones();
            }
        }
        if moving_overlap < min_overlap {
            continue;
        }
        let avg = total_bits as f32 / moving_overlap as f32;
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

/// Per-pair near-duplicate measurement. The visual alignment, plus a lazily
/// filled audio alignment (left `None` here; a caller corroborates borderline
/// pairs separately), plus which segment of each clip produced the winning
/// alignment.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PairScore {
    /// Average Hamming distance per scored frame of the winning visual
    /// alignment.
    pub visual_avg: f32,
    /// Number of frames that average was taken over.
    pub visual_overlap: usize,
    /// Audio corroboration, filled lazily by the caller for borderline pairs.
    /// `(avg_bits, overlap)`.
    pub audio: Option<(f32, usize)>,
    /// Index of the matched segment on the first (`a`) clip. Segment 0 is the
    /// head window; higher indices are tail windows.
    pub visual_seg_a: usize,
    /// Index of the matched segment on the second (`b`) clip.
    pub visual_seg_b: usize,
}

/// Score a single pair of hash sequences. `None` when either side is below the
/// hard overlap floor, or no shift meets the adaptive floor.
///
/// The effective floor is `min_overlap_frames` raised to at least
/// `min_overlap_hard_floor`, then capped at each sequence's length so short
/// clips can still match across their full length (but never below the hard
/// floor).
pub fn score_visual(a: &[u64], b: &[u64], params: &DedupParams) -> Option<PairScore> {
    if a.len() < params.min_overlap_hard_floor || b.len() < params.min_overlap_hard_floor {
        return None;
    }
    let effective_min = params
        .min_overlap_frames
        .max(params.min_overlap_hard_floor)
        .min(a.len())
        .min(b.len());
    let alignment = best_alignment(a, b, effective_min, params.motion_bits);
    if alignment.overlap < effective_min {
        return None;
    }
    Some(PairScore {
        visual_avg: alignment.avg_bits,
        visual_overlap: alignment.overlap,
        audio: None,
        visual_seg_a: 0,
        visual_seg_b: 0,
    })
}

/// Segment-aware visual score. A clip may be hashed as several independent
/// segments (a head window plus optional tail windows); two clips match
/// through any pairing of their segments. Returns the best (lowest-`avg_bits`)
/// [`score_visual`] over every segment pair, with `visual_seg_a` /
/// `visual_seg_b` set to the matched segment indices.
pub fn score_visual_segments(
    a: &[Vec<u64>],
    b: &[Vec<u64>],
    params: &DedupParams,
) -> Option<PairScore> {
    let mut best: Option<PairScore> = None;
    for (ia, sa) in a.iter().enumerate() {
        for (ib, sb) in b.iter().enumerate() {
            let Some(mut score) = score_visual(sa, sb, params) else {
                continue;
            };
            score.visual_seg_a = ia;
            score.visual_seg_b = ib;
            if best.is_none_or(|cur| score.visual_avg < cur.visual_avg) {
                best = Some(score);
            }
        }
    }
    best
}

/// A candidate duplicate edge between two items, oriented so `canonical` is the
/// lower-ranked (kept) item and `member` is the duplicate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DupeEdge<Id> {
    /// The duplicate item.
    pub member: Id,
    /// The kept (lower-ranked) item.
    pub canonical: Id,
    /// Average Hamming distance per scored frame of the winning alignment.
    pub avg_bits: f32,
    /// Number of frames that average was taken over.
    pub overlap_frames: usize,
}

impl<Id: Copy> DupeEdge<Id> {
    /// The `(canonical, member)` id pair, for feeding to
    /// [`crate::cluster::cluster_edges`].
    pub fn pair(&self) -> (Id, Id) {
        (self.canonical, self.member)
    }
}

/* Orient a pair into (canonical, member) by rank, tie-broken on Id order so the
choice is deterministic. Lower rank wins (becomes canonical). Shared with the
audio candidate finder. */
pub(crate) fn orient<Id: Copy + Eq + Hash + Ord>(
    id_a: Id,
    id_b: Id,
    ranks: &HashMap<Id, u64>,
) -> (Id, Id) {
    let rank_a = ranks.get(&id_a).copied().unwrap_or(u64::MAX);
    let rank_b = ranks.get(&id_b).copied().unwrap_or(u64::MAX);
    match rank_a.cmp(&rank_b) {
        std::cmp::Ordering::Less => (id_a, id_b),
        std::cmp::Ordering::Greater => (id_b, id_a),
        std::cmp::Ordering::Equal => {
            if id_a < id_b {
                (id_a, id_b)
            } else {
                (id_b, id_a)
            }
        }
    }
}

/// All candidate duplicate edges over an all-pairs comparison of `hashes`.
///
/// This is the in-memory, small-N path: every pair of items is aligned and
/// kept if it clears the overlap floor and sits at or under
/// `params.threshold_bits`. (A large corpus uses an index to generate
/// candidates instead; this crate provides the per-pair primitives for that —
/// see [`score_visual`].)
///
/// `ranks` decides which item of a matched pair is canonical (lower rank wins;
/// missing ranks sort last). `excluded` holds pairs a caller has marked as
/// "not duplicates", stored Id-ordered (lower first); they are skipped before
/// any comparison.
///
/// Generic over `V: AsRef<[u64]>` so a caller can pass owned `Vec<u64>` or
/// borrowed `&Vec<u64>` without cloning.
pub fn find_candidates<Id, V>(
    hashes: &HashMap<Id, V>,
    ranks: &HashMap<Id, u64>,
    excluded: &HashSet<(Id, Id)>,
    params: &DedupParams,
) -> Vec<DupeEdge<Id>>
where
    Id: Copy + Eq + Hash + Ord,
    V: AsRef<[u64]>,
{
    // Sort by id so the edge order (and any downstream clustering order) is
    // reproducible rather than dependent on HashMap iteration order.
    let mut entries: Vec<(Id, &[u64])> = hashes.iter().map(|(k, v)| (*k, v.as_ref())).collect();
    entries.sort_unstable_by_key(|(id, _)| *id);
    let mut out = Vec::new();
    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            let (id_a, ha) = entries[i];
            let (id_b, hb) = entries[j];
            let pair = if id_a < id_b {
                (id_a, id_b)
            } else {
                (id_b, id_a)
            };
            if excluded.contains(&pair) {
                continue;
            }
            if ha.len() < params.min_overlap_hard_floor || hb.len() < params.min_overlap_hard_floor
            {
                continue;
            }
            let effective_min = params
                .min_overlap_frames
                .max(params.min_overlap_hard_floor)
                .min(ha.len())
                .min(hb.len());
            let alignment = best_alignment(ha, hb, effective_min, params.motion_bits);
            if alignment.overlap < effective_min || alignment.avg_bits > params.threshold_bits {
                continue;
            }
            let (canonical, member) = orient(id_a, id_b, ranks);
            out.push(DupeEdge {
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

    // A sequence whose every consecutive frame differs by many bits, so every
    // frame clears any sane motion gate.
    fn moving_seq(n: u64) -> Vec<u64> {
        (0..n)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect()
    }

    #[test]
    fn motion_gate_off_reproduces_plain_average() {
        // A static all-zero run with one flipped bit: under the motion gate
        // (motion_bits >= 1) it scores nothing; with motion_bits = 0 it scores
        // the plain average 1/8.
        let a = vec![0u64; 8];
        let mut b = vec![0u64; 8];
        b[3] = 0x01;
        let plain = best_alignment(&a, &b, 8, 0);
        assert_eq!(plain.shift, 0);
        assert_eq!(plain.overlap, 8);
        assert!((plain.avg_bits - 1.0 / 8.0).abs() < 1e-6);

        let gated = best_alignment(&a, &b, 8, 2);
        assert_eq!(gated, Alignment::NO_MATCH);
    }

    #[test]
    fn order_independent_average() {
        let mv = moving_seq(50);
        let card = vec![0xABCD_1234u64; 40];
        let with_card: Vec<u64> = mv.iter().chain(card.iter()).copied().collect();
        let ab = best_alignment(&mv, &with_card, 30, 2);
        let ba = best_alignment(&with_card, &mv, 30, 2);
        assert_eq!(ab.avg_bits, ba.avg_bits);
        assert_eq!(ab.overlap, ba.overlap);
    }

    #[test]
    fn classify_distinguishes_subset_from_partial() {
        let a = moving_seq(60);
        let b = a[20..40].to_vec();
        let al = best_alignment(&a, &b, 8, 2);
        assert_eq!(al.classify(a.len(), b.len()), OverlapKind::Contains);
        let al2 = best_alignment(&b, &a, 8, 2);
        assert_eq!(al2.classify(b.len(), a.len()), OverlapKind::ContainedBy);
        let same = best_alignment(&a, &a, 8, 2);
        assert_eq!(same.classify(a.len(), a.len()), OverlapKind::Identical);
    }

    #[test]
    fn classify_uses_geometric_window_not_scored_frames() {
        // Two identical clips that include a long static run. The motion gate
        // scores fewer frames than the full length, but the clips are still
        // Identical — classification must use the geometric window, not the
        // (smaller) scored-frame count.
        let moving = moving_seq(40);
        let card = vec![0xABCDu64; 40];
        let clip: Vec<u64> = moving.iter().chain(card.iter()).copied().collect();
        let al = best_alignment(&clip, &clip, 30, 2);
        assert!(
            al.overlap < clip.len(),
            "the static run is excluded from scoring"
        );
        assert_eq!(al.classify(clip.len(), clip.len()), OverlapKind::Identical);
    }

    #[test]
    fn motion_gate_off_scores_single_frame() {
        // With the gate disabled, even a one-frame sequence is scored.
        let al = best_alignment(&[5u64], &[5u64], 1, 0);
        assert!(al.matched());
        assert_eq!(al.avg_bits, 0.0);
        assert_eq!(al.overlap, 1);
    }

    #[test]
    fn zero_min_overlap_is_coerced_and_finite() {
        // min_overlap = 0 must not produce a 0/0 NaN; it is treated as 1.
        let seq = moving_seq(40);
        let al = best_alignment(&seq, &seq, 0, 2);
        assert!(al.avg_bits.is_finite());
        assert_eq!(al.avg_bits, 0.0);
        assert!(al.overlap >= 1);
    }
}

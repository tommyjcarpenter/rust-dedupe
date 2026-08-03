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
//! Setting `motion_bits = 0` disables the gate: every frame is scored
//! (including a lone single-frame sequence), which reproduces the plain
//! average-over-all-overlapping-frames behavior. The two regimes are otherwise
//! the same code path.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::config::DedupParams;

/// The result of aligning two hash sequences: the winning shift, the average
/// Hamming distance per scored (moving) frame, and how many frames that
/// average was taken over.
///
/// A positive `shift` means the second sequence (`b`) starts later in the first
/// sequence's (`a`) frame of reference — i.e. `b` is `a` with its front
/// trimmed. A negative shift means `b` starts earlier — `a` is `b` with its
/// front trimmed.
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

/* Build a per-element "moving" mask: element k is moving if it differs from EITHER
neighbor by at least `motion_bits`. Using "either neighbor" keeps the boundary
element of a moving run counted (no off-by-one that would drop a minimum-length
clip below the floor), while the interior of a frozen run counts zero.

`motion_bits == 0` disables the gate entirely: every element is scored, including
a single-element sequence (which has no neighbor to compare against).

Generic over the element so the audio path shares it: a frozen run means the same
thing on either signal, a repeated frame or silence fingerprinting to a constant. */
pub(crate) fn moving_mask<T>(
    seq: &[T],
    motion_bits: u32,
    dist: impl Fn(&T, &T) -> u32,
) -> Vec<bool> {
    if motion_bits == 0 {
        return vec![true; seq.len()];
    }
    (0..seq.len())
        .map(|k| {
            let prev = k > 0 && dist(&seq[k], &seq[k - 1]) >= motion_bits;
            let next = k + 1 < seq.len() && dist(&seq[k], &seq[k + 1]) >= motion_bits;
            prev || next
        })
        .collect()
}

/// Hamming distance between two 64-bit frame hashes.
pub(crate) fn hamming64_dist(a: &u64, b: &u64) -> u32 {
    (a ^ b).count_ones()
}

/* Score one aligned window: `(elements scored, total bits)`. `masks` is `None` when the gate is
off, and is branched on once per window rather than once per element — an all-true mask is worth
nothing per element and costs the ungated arm its vectorization over an O(len_a * len_b) sweep.
Shared by both signals so the fast path can't go missing on one of them. */
pub(crate) fn score_window<T>(
    a: &[T],
    b: &[T],
    masks: Option<(&[bool], &[bool])>,
    dist: impl Fn(&T, &T) -> u32,
) -> (usize, u64) {
    // u64 accumulator: a long overlap can sum past u32::MAX bits.
    let Some((ma, mb)) = masks else {
        let total: u64 = a.iter().zip(b).map(|(x, y)| u64::from(dist(x, y))).sum();
        return (a.len(), total);
    };
    let mut scored = 0usize;
    let mut total = 0u64;
    for (((x, y), &am), &bm) in a.iter().zip(b).zip(ma).zip(mb) {
        if am || bm {
            scored += 1;
            total += u64::from(dist(x, y));
        }
    }
    (scored, total)
}

/// A well-matching stretch found inside an alignment whose overall distance may be poor.
///
/// [`Alignment`] answers "are these the same clip", averaging over the whole intersection. That
/// is the wrong question when two clips share a scene and then diverge: the shared part is
/// averaged with the part that isn't, and a real overlap vanishes into the mean. This answers
/// "do these clips share footage, and which part".
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RunMatch {
    /// Offset of `b` relative to `a`, with the same meaning and sign as [`Alignment::shift`].
    pub shift: i32,
    /// Where the run starts in `a`.
    pub run_start_a: usize,
    /// Where the run starts in `b`.
    pub run_start_b: usize,
    /// Elements the run spans. Larger than `run_scored` when the motion gate skipped
    /// something inside it.
    pub run_len: usize,
    /// Elements actually scored in the run — always the effective overlap floor, since the run
    /// is defined as the best window holding exactly that many.
    pub run_scored: usize,
    /// Average distance over the run's scored elements: whether this stretch is shared content.
    pub run_avg_bits: f32,
    /// Average over the WHOLE intersection AT THIS SHIFT — the per-shift number, not
    /// [`best_alignment`]'s result. That minimizes over every shift, so its `avg_bits` is a lower
    /// bound on this and usually comes from a different shift entirely.
    ///
    /// Carried so a caller can see the contrast that motivates the run scan at all: a low
    /// `run_avg_bits` beside a high `full_avg_bits` is exactly a partial overlap.
    pub full_avg_bits: f32,
}

/// How much two sequences actually share, walked out from a [`RunMatch`] seed.
///
/// A [`RunMatch`] is the best window holding exactly `run_scored` elements, so it answers "is
/// there a match here" — which is what a verdict needs, and all it needs. It cannot answer "how
/// much is shared": every overlap reports one window, so two sequences sharing two minutes and two
/// sharing nine seconds come back the same length, and the reported start sits wherever the match
/// was easiest to prove rather than where the content began matching.
///
/// Produced by [`shared_span`] / [`shared_audio_span`], which take a seed and the sequences it
/// came from. Separate from the scan because it is a separate question: a caller deciding only
/// whether a pair is a duplicate never needs it and should not pay for the walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SharedSpan {
    /// Where the shared stretch begins in `a`, at or before the seed's `run_start_a`.
    pub start_a: usize,
    /// Where the shared stretch begins in `b`, at or before the seed's `run_start_b`.
    pub start_b: usize,
    /// Elements the stretch spans, at or above the seed's `run_len`.
    pub len: usize,
}

impl SharedSpan {
    /// One past the last element of the stretch in `a`.
    pub fn end_a(&self) -> usize {
        self.start_a + self.len
    }

    /// One past the last element of the stretch in `b`.
    pub fn end_b(&self) -> usize {
        self.start_b + self.len
    }
}

/* The best-matching run at one offset: `(start, span, avg)`, or `None` when the window never
holds `w` scored elements. Two pointers over the overlap maintaining a window with exactly `w` of
them, so the whole sweep stays O(len_a * len_b) — the same cost as the alignment it sits beside.

Exactly `w`, not "at least": minimizing a mean over variable-length windows is a different and
far more expensive problem, and `w` is already the caller's statement of how much evidence a
match needs. Leading unscored elements are trimmed so the reported span is the tightest one
holding the run. */
fn best_run_at<T>(
    a_win: &[T],
    b_win: &[T],
    masks: Option<(&[bool], &[bool])>,
    w: usize,
    dist: &impl Fn(&T, &T) -> u32,
) -> Option<(usize, usize, f32)> {
    let scored_at = |i: usize| masks.is_none_or(|(ma, mb)| ma[i] || mb[i]);
    let dist_at = |i: usize| u64::from(dist(&a_win[i], &b_win[i]));
    let mut best: Option<(usize, usize, f32)> = None;
    let mut lo = 0usize;
    let mut scored = 0usize;
    let mut sum = 0u64;
    for hi in 0..a_win.len() {
        if scored_at(hi) {
            scored += 1;
            sum += dist_at(hi);
        }
        while scored > w {
            if scored_at(lo) {
                scored -= 1;
                sum -= dist_at(lo);
            }
            lo += 1;
        }
        if scored < w {
            continue;
        }
        while lo < hi && !scored_at(lo) {
            lo += 1;
        }
        let avg = (sum as f64 / w as f64) as f32;
        if best.is_none_or(|(_, _, prev)| avg < prev) {
            best = Some((lo, hi + 1 - lo, avg));
        }
    }
    best
}

/* Consecutive failing elements the walk tolerates before it calls the content diverged.

One bad element is not a boundary: a compression artifact, a dropped frame, a transient in the
audio. Requiring a clean break of more than this many in a row stops the walk at a real divergence
rather than at noise. The frontier is the last element that MATCHED, so tolerated misses that never
recover extend nothing — they only buy the walk a chance to continue. */
const MAX_CONSECUTIVE_MISSES: usize = 2;

/* The per-element bar for "this is still the same content", from the seed's own quality.

Absolute bars do not transfer: a pristine copy runs a couple of bits apart and a re-encode ten, so
a bar tight enough for the first cuts the second short and a bar loose enough for the second walks
the first into unrelated footage. Two unrelated elements sit around `width/2` — half the bits
differ by chance — so the midpoint between the seed's average and that baseline separates the two
cases without a tuned constant: still closer to the run than to chance. */
fn extend_tolerance(run_avg_bits: f32, width_bits: u32) -> u32 {
    let chance = width_bits as f32 / 2.0;
    if !run_avg_bits.is_finite() || run_avg_bits >= chance {
        return 0;
    }
    ((run_avg_bits + chance) / 2.0) as u32
}

/* How far the shared stretch really reaches, walking out from the seed along the fixed shift.

The seed is the best window of exactly `w` scored elements, so it says where a match was easiest to
prove, not where the footage started matching. Both clips advance together — the shift is fixed
inside a shared stretch — so this is a linear walk from the seed's edges, bounded by whichever
sequence ends first. Cost is the length of what it finds, against the O(len_a * len_b) scan that
found the seed.

Gated elements (a static run, digital silence) carry no signal, so they are stepped over without
counting as a match or a miss. They cannot run the walk away: the frontier only advances on an
element that matched, so coasting through a silence and then finding nothing leaves the frontier
where it was. Coasting through a silence and then finding MORE matching content is the correct
answer — content that matches on both sides of a silence at one shift is one stretch. */
fn extend_run<T>(
    a: &[T],
    b: &[T],
    masks: Option<&(Vec<bool>, Vec<bool>)>,
    seed: (usize, usize, usize),
    tolerance: u32,
    dist: impl Fn(&T, &T) -> u32,
) -> (usize, usize, usize) {
    let (seed_a, seed_b, seed_len) = seed;
    let scored_at = |ia: usize, ib: usize| masks.is_none_or(|(ma, mb)| ma[ia] || mb[ib]);
    let holds = |ia: usize, ib: usize| dist(&a[ia], &b[ib]) <= tolerance;

    let (mut start_a, mut start_b) = (seed_a, seed_b);
    let (mut ia, mut ib) = (seed_a, seed_b);
    let mut misses = 0usize;
    while ia > 0 && ib > 0 {
        ia -= 1;
        ib -= 1;
        if !scored_at(ia, ib) {
            continue;
        }
        if holds(ia, ib) {
            start_a = ia;
            start_b = ib;
            misses = 0;
        } else {
            misses += 1;
            if misses > MAX_CONSECUTIVE_MISSES {
                break;
            }
        }
    }

    // Exclusive, in a's index space; b tracks it at the same shift.
    let mut end_a = seed_a + seed_len;
    let (mut ia, mut ib) = (seed_a + seed_len, seed_b + seed_len);
    let mut misses = 0usize;
    while ia < a.len() && ib < b.len() {
        if scored_at(ia, ib) {
            if holds(ia, ib) {
                end_a = ia + 1;
                misses = 0;
            } else {
                misses += 1;
                if misses > MAX_CONSECUTIVE_MISSES {
                    break;
                }
            }
        }
        ia += 1;
        ib += 1;
    }

    (start_a, start_b, end_a - start_a)
}

/* Shared core of the run scan. Same offset range and floors as `best_alignment`, but scoring the
best window at each offset instead of the whole overlap. */
pub(crate) fn best_run_generic<T>(
    a: &[T],
    b: &[T],
    min_overlap: usize,
    hard_floor: usize,
    motion_bits: u32,
    dist: impl Fn(&T, &T) -> u32 + Copy,
) -> Option<RunMatch> {
    if a.len() < hard_floor || b.len() < hard_floor {
        return None;
    }
    let w = min_overlap.max(hard_floor).min(a.len()).min(b.len()).max(1);
    if a.is_empty() || b.is_empty() || a.len() > i32::MAX as usize || b.len() > i32::MAX as usize {
        return None;
    }
    let max_pos: i32 = a.len() as i32 - w as i32;
    let max_neg: i32 = w as i32 - b.len() as i32;
    if max_pos < max_neg {
        return None;
    }
    let masks = masks_for(a, b, motion_bits, dist);
    let mut best: Option<RunMatch> = None;
    for shift in max_neg..=max_pos {
        let (a_start, b_start) = if shift >= 0 {
            (shift as usize, 0)
        } else {
            (0, (-shift) as usize)
        };
        let overlap = (a.len() - a_start).min(b.len() - b_start);
        if overlap < w {
            continue;
        }
        let a_win = &a[a_start..a_start + overlap];
        let b_win = &b[b_start..b_start + overlap];
        let win_masks = masks.as_ref().map(|(ma, mb)| {
            (
                &ma[a_start..a_start + overlap],
                &mb[b_start..b_start + overlap],
            )
        });
        let Some((run_off, run_len, run_avg)) = best_run_at(a_win, b_win, win_masks, w, &dist)
        else {
            continue;
        };
        if best.is_some_and(|prev| run_avg >= prev.run_avg_bits) {
            continue;
        }
        let (scored, total) = score_window(a_win, b_win, win_masks, dist);
        best = Some(RunMatch {
            shift,
            run_start_a: a_start + run_off,
            run_start_b: b_start + run_off,
            run_len,
            run_scored: w,
            run_avg_bits: run_avg,
            full_avg_bits: if scored == 0 {
                f32::MAX
            } else {
                (total as f64 / scored as f64) as f32
            },
        });
    }
    best
}

/* Shared core of the outward walk. Rebuilds the motion masks the scan already built once, which
is O(len) against the O(len_a * len_b) sweep that produced the seed — the price of keeping this a
separate call, and small enough that it buys the caller the choice of not walking at all. */
pub(crate) fn shared_span_generic<T>(
    a: &[T],
    b: &[T],
    run: &RunMatch,
    motion_bits: u32,
    width_bits: u32,
    dist: impl Fn(&T, &T) -> u32 + Copy,
) -> SharedSpan {
    let seed = (run.run_start_a, run.run_start_b, run.run_len);
    if run.run_start_a + run.run_len > a.len() || run.run_start_b + run.run_len > b.len() {
        // A seed from different sequences than the ones handed in. Report it unchanged rather
        // than indexing out of bounds on a caller's mix-up.
        return SharedSpan {
            start_a: run.run_start_a,
            start_b: run.run_start_b,
            len: run.run_len,
        };
    }
    let masks = masks_for(a, b, motion_bits, dist);
    let (start_a, start_b, len) = extend_run(
        a,
        b,
        masks.as_ref(),
        seed,
        extend_tolerance(run.run_avg_bits, width_bits),
        dist,
    );
    SharedSpan {
        start_a,
        start_b,
        len,
    }
}

/// How much of two frame-hash sequences is really shared, walking out from a [`RunMatch`] seed
/// until the content diverges. See [`SharedSpan`].
///
/// `run` must be the result of [`best_matching_run`] on these same two sequences, and `params` the
/// same ones — the walk reads the seed's own average to decide what still counts as matching.
pub fn shared_span(a: &[u64], b: &[u64], run: &RunMatch, params: &DedupParams) -> SharedSpan {
    shared_span_generic(a, b, run, params.motion_bits, u64::BITS, hamming64_dist)
}

/// The best-matching run between two frame-hash sequences, for finding a shared stretch inside
/// clips that are not the same clip end to end. See [`RunMatch`].
///
/// Uses the same floors and motion gate as [`score_visual`], so a run is held to the same
/// evidence bar as an alignment; only the span it averages over differs.
pub fn best_matching_run(a: &[u64], b: &[u64], params: &DedupParams) -> Option<RunMatch> {
    best_run_generic(
        a,
        b,
        params.min_overlap_frames,
        params.min_overlap_hard_floor,
        params.motion_bits,
        hamming64_dist,
    )
}

/// Both sequences' moving masks, or `None` when the gate is off. Built once per pair.
pub(crate) fn masks_for<T>(
    a: &[T],
    b: &[T],
    motion_bits: u32,
    dist: impl Fn(&T, &T) -> u32 + Copy,
) -> Option<(Vec<bool>, Vec<bool>)> {
    (motion_bits > 0).then(|| {
        (
            moving_mask(a, motion_bits, dist),
            moving_mask(b, motion_bits, dist),
        )
    })
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
    let masks = masks_for(a, b, motion_bits, hamming64_dist);
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
        let a_win = &a[a_start..a_start + overlap];
        let b_win = &b[b_start..b_start + overlap];
        let win_masks = masks.as_ref().map(|(ma, mb)| {
            (
                &ma[a_start..a_start + overlap],
                &mb[b_start..b_start + overlap],
            )
        });
        let (moving_overlap, total_bits) = score_window(a_win, b_win, win_masks, hamming64_dist);
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

    /* Params with the gate off and a 10-element floor, so the run tests state their own bar
    rather than inheriting the shipped defaults. */
    fn run_params(min_overlap: usize) -> DedupParams {
        DedupParams {
            min_overlap_frames: min_overlap,
            min_overlap_hard_floor: min_overlap,
            motion_bits: 0,
            ..DedupParams::default()
        }
    }

    #[test]
    fn a_shared_opening_is_found_though_the_clips_diverge() {
        /* The case the run scan exists for. Both clips open with the same 20 elements and then
        continue with unrelated content, so they stick out in the SAME direction and the
        intersection outlasts the shared part. The plain alignment averages 20 matching elements
        with 40 unrelated ones and sees nothing; the run scan finds the opening. */
        let shared = moving_seq(20);
        let a: Vec<u64> = shared.iter().copied().chain(moving_seq(40)).collect();
        let b: Vec<u64> = shared
            .iter()
            .copied()
            .chain(moving_seq(40).iter().map(|h| !h))
            .collect();
        let p = run_params(10);

        let full = best_alignment(&a, &b, p.min_overlap_frames, p.motion_bits);
        assert!(
            full.avg_bits > 10.0,
            "the plain average is diluted past any threshold: {}",
            full.avg_bits
        );

        let run = best_matching_run(&a, &b, &p).expect("a run is found");
        assert_eq!(run.shift, 0);
        assert_eq!(run.run_start_a, 0);
        assert_eq!(run.run_start_b, 0);
        assert_eq!(run.run_avg_bits, 0.0, "the shared opening is exact");
        assert_eq!(run.run_scored, 10);
        /* The contrast in one struct, which is the whole point of carrying both: at the run's
        own offset the whole-overlap mean is still diluted past any threshold. Not compared
        against `full` above — `best_alignment` minimizes over every offset and can win at a
        different one, so its number is a floor, not the value at this offset. */
        assert!(
            run.full_avg_bits > 10.0,
            "the same offset is diluted end to end: {}",
            run.full_avg_bits
        );
        assert!(run.full_avg_bits >= full.avg_bits);
    }

    #[test]
    fn the_run_is_located_in_the_middle_of_both_clips() {
        // A shared stretch buried inside both sides — the interior case a plain alignment
        // cannot isolate, because the intersection is pinned to the shorter clip's length.
        let shared = moving_seq(15);
        let a: Vec<u64> = moving_seq(25)
            .iter()
            .map(|h| !h)
            .chain(shared.iter().copied())
            .chain(moving_seq(25).iter().map(|h| h.rotate_left(7)))
            .collect();
        let b: Vec<u64> = moving_seq(10)
            .iter()
            .map(|h| h.rotate_left(19))
            .chain(shared.iter().copied())
            .chain(moving_seq(30).iter().map(|h| h.rotate_left(3)))
            .collect();
        let run = best_matching_run(&a, &b, &run_params(12)).expect("a run is found");
        assert_eq!(run.run_avg_bits, 0.0);
        assert_eq!(run.run_start_a, 25);
        assert_eq!(run.run_start_b, 10);
        assert_eq!(run.shift, 15, "b sits 15 elements later in a's frame");
    }

    #[test]
    fn identical_clips_report_a_perfect_run_and_a_perfect_whole() {
        let seq = moving_seq(40);
        let run = best_matching_run(&seq, &seq, &run_params(10)).expect("a run is found");
        assert_eq!(run.run_avg_bits, 0.0);
        assert_eq!(
            run.full_avg_bits, 0.0,
            "nothing to contrast when it all matches"
        );
        assert_eq!(run.shift, 0);
    }

    #[test]
    fn unrelated_clips_report_a_run_that_is_still_far() {
        /* The scan always returns its best window, so the DISTANCE is what rejects a pair, not
        the absence of a result. A caller that treated `Some` as a match would flag everything. */
        let a = moving_seq(60);
        let b: Vec<u64> = moving_seq(60).iter().map(|h| h.rotate_left(31)).collect();
        let run = best_matching_run(&a, &b, &run_params(10)).expect("still returns its best");
        assert!(
            run.run_avg_bits > 10.0,
            "unrelated content stays far even at its best window: {}",
            run.run_avg_bits
        );
    }

    #[test]
    fn a_clip_below_the_hard_floor_has_no_run() {
        let short = moving_seq(5);
        let long = moving_seq(60);
        assert_eq!(best_matching_run(&short, &long, &run_params(10)), None);
        assert_eq!(best_matching_run(&long, &short, &run_params(10)), None);
    }

    #[test]
    fn the_motion_gate_keeps_a_static_run_from_becoming_a_match() {
        /* Without the gate, two clips sharing only a black end card align at 0 bits over it and
        the run scan reports a perfect match — the same false positive the gate exists to stop on
        the alignment path, and more dangerous here because the scan is looking for exactly that
        shape. */
        let card = vec![0xABCD_1234u64; 40];
        let a: Vec<u64> = moving_seq(40).iter().copied().chain(card.clone()).collect();
        let b: Vec<u64> = moving_seq(40)
            .iter()
            .map(|h| !h)
            .chain(card.iter().copied())
            .collect();

        let ungated = best_matching_run(&a, &b, &run_params(10)).expect("finds the card");
        assert_eq!(
            ungated.run_avg_bits, 0.0,
            "ungated, the shared card is perfect"
        );

        let gated = best_matching_run(
            &a,
            &b,
            &DedupParams {
                motion_bits: 2,
                ..run_params(10)
            },
        );
        let far = gated.is_none_or(|r| r.run_avg_bits > 10.0);
        assert!(far, "gated, the card cannot carry the run: {gated:?}");
    }

    /* A shared stretch of `shared` elements sitting `lead_a` / `lead_b` into two otherwise
    unrelated sequences. The leads differ so the shift is non-zero and the walk has to move both
    sides together to stay on the content. */
    fn shared_stretch(
        lead_a: usize,
        lead_b: usize,
        shared: usize,
        tail: usize,
    ) -> (Vec<u64>, Vec<u64>) {
        let common = moving_seq(shared as u64);
        let mut a: Vec<u64> = moving_seq(lead_a as u64).iter().map(|x| !x).collect();
        a.extend(common.iter().copied());
        a.extend(
            moving_seq(tail as u64)
                .iter()
                .map(|x| x ^ 0x5555_5555_5555_5555),
        );
        let mut b: Vec<u64> = moving_seq(lead_b as u64)
            .iter()
            .map(|x| x ^ 0x0F0F)
            .collect();
        b.extend(common.iter().copied());
        b.extend(
            moving_seq(tail as u64)
                .iter()
                .map(|x| x ^ 0xAAAA_AAAA_AAAA_AAAA),
        );
        (a, b)
    }

    /* The same shared stretch, but only its middle third is a pristine copy — the outer thirds
    carry a few flipped bits, as a re-encode does. That is what puts the seed strictly INSIDE the
    stretch: the window with the lowest average is the clean middle, so the reported start is late
    and the reported end early, exactly the real-footage behavior. A perfectly-matching stretch
    ties at zero everywhere and the first window wins, which would hide a missing backward walk. */
    fn shared_stretch_clean_middle(
        lead_a: usize,
        lead_b: usize,
        shared: usize,
        tail: usize,
    ) -> (Vec<u64>, Vec<u64>) {
        let (a, mut b) = shared_stretch(lead_a, lead_b, shared, tail);
        let third = shared / 3;
        for k in 0..shared {
            if k < third || k >= shared - third {
                b[lead_b + k] ^= 0b111;
            }
        }
        (a, b)
    }

    #[test]
    fn the_walk_recovers_the_whole_shared_stretch_not_the_seed() {
        /* The seed is the best window holding exactly `min_overlap` elements, so on a stretch far
        longer than the window it lands somewhere inside and says nothing about the extent. What is
        wanted is where the footage starts and stops matching. */
        let (a, b) = shared_stretch_clean_middle(40, 12, 90, 30);
        let r = best_matching_run(&a, &b, &run_params(10)).expect("a 90-element stretch matches");
        assert_eq!(r.run_scored, 10, "the seed is one window");
        assert!(
            r.run_start_a > 40 && r.run_start_a + r.run_len < 130,
            "the seed must sit strictly inside the stretch or this proves nothing: {r:?}"
        );
        let span = shared_span(&a, &b, &r, &run_params(10));
        assert_eq!(
            (span.start_a, span.start_b, span.len),
            (40, 12, 90),
            "the walk recovers the real edges on both sides"
        );
        assert_eq!((span.end_a(), span.end_b()), (130, 102));
    }

    #[test]
    fn the_walk_stops_where_the_content_diverges() {
        // Everything outside the stretch is unrelated, so neither edge may leak into it. The seed
        // starts inside, so both walks travel before they have to stop at the right place.
        let (a, b) = shared_stretch_clean_middle(25, 25, 40, 25);
        let r = best_matching_run(&a, &b, &run_params(10)).expect("a 40-element stretch matches");
        let span = shared_span(&a, &b, &r, &run_params(10));
        assert_eq!(span.start_a, 25, "did not walk back into unrelated lead");
        assert_eq!(span.len, 40, "did not walk forward into unrelated tail");
    }

    #[test]
    fn the_walk_stops_at_the_sequence_edge() {
        // A stretch running to the very head of both clips: the walk has to end at 0, not underflow.
        let (a, b) = shared_stretch(0, 0, 50, 20);
        let r = best_matching_run(&a, &b, &run_params(10)).expect("matches");
        let span = shared_span(&a, &b, &r, &run_params(10));
        assert_eq!((span.start_a, span.start_b), (0, 0));
        assert_eq!(span.len, 50);
    }

    #[test]
    fn one_damaged_element_does_not_end_the_stretch() {
        /* A single bad element is noise, not a boundary — a compression artifact or a dropped
        frame. Requiring a clean break of several in a row is what keeps the walk from stopping on
        it, and the reported extent has to span the damage. */
        let (a, mut b) = shared_stretch(20, 20, 60, 20);
        b[20 + 45] = !b[20 + 45];
        let r = best_matching_run(&a, &b, &run_params(10)).expect("matches");
        let span = shared_span(&a, &b, &r, &run_params(10));
        assert_eq!(span.start_a, 20);
        assert_eq!(span.len, 60, "walked straight past the damaged element");
    }

    #[test]
    fn the_extent_is_never_shorter_than_the_seed() {
        /* The invariant every consumer leans on: the seed is a stretch that matched, so the walk
        can only ever agree with it or find more. Checked on a stretch barely longer than the
        window, where there is almost nothing to extend into. */
        let (a, b) = shared_stretch(15, 8, 12, 15);
        let r = best_matching_run(&a, &b, &run_params(10)).expect("matches");
        let span = shared_span(&a, &b, &r, &run_params(10));
        assert!(span.len >= r.run_len, "{span:?} vs {r:?}");
        assert!(span.start_a <= r.run_start_a, "{span:?} vs {r:?}");
        assert!(span.start_b <= r.run_start_b, "{span:?} vs {r:?}");
    }

    #[test]
    fn the_walk_steps_over_a_gated_run_to_reach_content_beyond_it() {
        /* A static run carries no signal, so it neither extends the stretch nor ends it. Content
        matching on both sides of one at the same offset is one stretch, and reporting two would
        make the caller cut a clip in the middle of footage it shares. */
        let common = moving_seq(30);
        let card = [0xABCD_1234u64; 25];
        let build = |lead: u64, salt: u64| {
            let mut v: Vec<u64> = moving_seq(lead).iter().map(|x| x ^ salt).collect();
            v.extend(common.iter().copied());
            v.extend(card.iter().copied());
            v.extend(common.iter().map(|x| x ^ 0x00FF));
            v
        };
        let a = build(10, 0xF0F0_0000_0000_0000);
        let b = build(10, 0x0F0F_0000_0000_0000);
        let params = DedupParams {
            motion_bits: 2,
            ..run_params(10)
        };
        let r = best_matching_run(&a, &b, &params).expect("the moving content matches");
        let span = shared_span(&a, &b, &r, &params);
        assert!(
            span.len >= 30 + 25 + 30,
            "the stretch has to span the static run, got {span:?}"
        );
    }

    #[test]
    fn a_seed_from_other_sequences_is_reported_unchanged() {
        // A caller mixing up which sequences a seed came from gets the seed back, not a panic.
        let (a, b) = shared_stretch(10, 10, 40, 10);
        let r = best_matching_run(&a, &b, &run_params(10)).expect("matches");
        let span = shared_span(&a[..12], &b[..12], &r, &run_params(10));
        assert_eq!(
            (span.start_a, span.start_b, span.len),
            (r.run_start_a, r.run_start_b, r.run_len)
        );
    }

    #[test]
    fn the_tolerance_sits_between_the_seed_and_chance() {
        /* Derived from the seed rather than fixed, so a pristine copy and a re-encode each get a
        bar suited to their own noise floor. Both stay well under chance, which is half the width. */
        assert_eq!(extend_tolerance(0.0, 64), 16, "pristine: half of chance");
        assert_eq!(
            extend_tolerance(8.0, 64),
            20,
            "a noisy re-encode gets more room"
        );
        assert_eq!(extend_tolerance(0.0, 32), 8, "narrower hashes scale down");
        assert_eq!(extend_tolerance(2.0, 32), 9);
        // A seed no better than chance is not a match; nothing should extend from it.
        assert_eq!(extend_tolerance(16.0, 32), 0);
        assert_eq!(extend_tolerance(f32::NAN, 64), 0);
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

//! Tunable parameters for matching and clustering.
//!
//! Every value that a consumer might reasonably want to tune lives here, in a
//! single struct with sensible defaults, rather than being baked into the
//! algorithm as a constant. The one exception is the pixel-side sampling
//! geometry (`frame_hash::SAMPLE_W` / `SAMPLE_H` / `SAMPLE_BYTES` /
//! `SAMPLE_FPS`): those define the exact bytes fed to the difference hash, so
//! changing them would make every previously-computed hash incomparable. They
//! are fixed constants, not parameters.
//!
//! The defaults are the more conservative, motion-gated values: a 60-second
//! sample window, a 10-bit average-Hamming visual threshold, and the motion
//! gate enabled (`motion_bits = 2`). A consumer that wants the plain
//! average-over-all-overlapping-frames behavior sets `motion_bits = 0`, which
//! disables the gate entirely.

/// Matching and clustering tunables. Construct with [`DedupParams::default`]
/// and override individual fields, or build one explicitly.
///
/// Most visual fields are consumed by [`crate::align`] and most audio fields by
/// [`crate::audio`]; `sample_window_secs` is a caller-side extraction parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DedupParams {
    /// Seconds of content sampled per clip. Not consumed by the matching primitives;
    /// callers use this to decide how many hashes/sub-fingerprints to extract and
    /// store per item (e.g., the window length they pass to their extractor).
    pub sample_window_secs: u32,

    /// Average per-frame Hamming distance at or below which a visual alignment is close enough to be considered
    /// at all. Roughly: bits-flipped per 64-bit frame hash.
    ///
    /// Under [`crate::classify`] this is the outer ceiling rather than the admission line: a pair inside it is a
    /// duplicate only if it is also near-identical, or its audio agrees. Nothing past it is ever a duplicate.
    pub threshold_bits: f32,

    /// Average visual distance at or below which a match is "near-identical" and stands on the visual signal
    /// alone, without audio corroboration.
    ///
    /// Read by [`crate::classify`] and [`crate::needs_audio_corroboration`] only. The matching primitives
    /// ([`crate::score_visual`], [`crate::find_candidates`]) do not consult it, so a consumer that skips the
    /// policy layer is unaffected by it.
    pub near_identical_visual_bits: f32,

    /// Frame-to-frame hash change at or above which content is considered to
    /// be MOVING. A static run (a repeated watermark or end card: the same
    /// frame over and over) moves by ~0 bits and is excluded from scoring, so
    /// two clips sharing only such a run do not falsely match.
    ///
    /// Set to `0` to disable the motion gate entirely, scoring the plain
    /// average over all overlapping frames.
    pub motion_bits: u32,

    /// Target minimum number of overlapping frames a visual alignment must
    /// cover. Adapts down per pair toward `min_overlap_hard_floor` for short
    /// clips, but never below it.
    pub min_overlap_frames: usize,

    /// Hard floor below which a visual match is refused regardless of clip
    /// length. Below this, the sliding search finds false positives reliably
    /// enough that no threshold rescues precision.
    pub min_overlap_hard_floor: usize,

    /// Average per-sub-fingerprint Hamming distance at or below which an audio
    /// alignment is taken as a match. Out of 32 bits.
    pub audio_threshold_bits: f32,

    /// Target minimum number of overlapping audio sub-fingerprints. Adapts
    /// down per pair toward `audio_min_overlap_hard_floor`, never below it.
    pub audio_min_overlap: usize,

    /// Hard floor below which an audio match is refused regardless of length.
    pub audio_min_overlap_hard_floor: usize,

    /// The audio analogue of `motion_bits`. Silence fingerprints to a constant run, so
    /// without the gate two silent stretches align at ~0 bits and match anything. `0`
    /// (the default) disables it.
    pub audio_motion_bits: u32,

    /// Average audio distance at or below which audio admits a pair ON ITS OWN, with no
    /// visual agreement and no ceiling bounding it.
    ///
    /// A frame hash cannot see through a reframe — a de-pillarboxed crop and a smaller
    /// copy of the same clip read ~20 of 64 bits apart — so for those pairs the ceiling
    /// bounds noise. Set this far tighter than `audio_threshold_bits`: admitting a pair
    /// alone is a higher bar than corroborating one already in range. Clips can share a
    /// soundtrack over unrelated footage, so a consumer that DELETES should treat this
    /// tier as review-not-delete.
    ///
    /// [`f32::NEG_INFINITY`] (the default) disables it; any non-finite bound fails closed.
    pub audio_alone_bits: f32,
}

impl Default for DedupParams {
    fn default() -> Self {
        Self {
            sample_window_secs: 60,
            threshold_bits: 10.0,
            near_identical_visual_bits: 3.0,
            motion_bits: 2,
            min_overlap_frames: 30,
            min_overlap_hard_floor: 30,
            audio_threshold_bits: 3.0,
            audio_min_overlap: 30,
            audio_min_overlap_hard_floor: 30,
            audio_motion_bits: 0,
            audio_alone_bits: f32::NEG_INFINITY,
        }
    }
}

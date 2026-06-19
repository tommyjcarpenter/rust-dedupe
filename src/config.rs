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
/// The visual fields drive [`crate::align`]; the audio fields drive
/// [`crate::audio`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DedupParams {
    /// Seconds of content sampled per clip. Governs how many frames a newly
    /// ingested clip stores; a larger window lets the sliding search find
    /// matches with larger trim offsets between two clips' in-points.
    pub sample_window_secs: u32,

    /// Average per-frame Hamming distance at or below which a visual alignment
    /// is taken as a duplicate. Roughly: bits-flipped per 64-bit frame hash.
    pub threshold_bits: f32,

    /// Average visual distance at or below which two clips are taken as
    /// genuinely repeated content and flagged WITHOUT audio corroboration.
    /// Far stricter than `threshold_bits`; visually homogeneous footage reads
    /// in the single digits even between unrelated clips, so anything above
    /// this floor needs a second signal to become an edge.
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
        }
    }
}

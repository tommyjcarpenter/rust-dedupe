//! Perceptual near-duplicate detection primitives.
//!
//! This crate is the single, content-neutral home for a perceptual
//! de-duplication algorithm shared across projects. It owns the *pure*
//! algorithm — hashing, alignment, scoring, clustering — and, behind feature
//! flags, the `ffmpeg` extraction and image decoding. It does NOT own storage,
//! candidate indexing against a large corpus, or keep/delete orchestration:
//! those differ per consumer, which keep them and call these primitives.
//!
//! # What it does
//!
//! - **Video / clip dedup.** A clip is sampled into a sequence of 64-bit
//!   difference hashes ([`frame_hash::dhash_9x8`]). Two clips are compared by
//!   sliding one sequence against the other and taking the alignment with the
//!   lowest average Hamming distance per overlapping frame
//!   ([`align::best_alignment`]), which handles trim offsets. A motion gate
//!   stops two clips that share only a static end card from matching. An
//!   independent audio sub-fingerprint signal ([`audio`]) corroborates
//!   borderline visual matches.
//! - **Image dedup.** A 256-bit difference hash ([`image_hash::ImageHash`])
//!   with a multi-probe pigeonhole index for fast lookup.
//! - **Clustering.** Pairwise edges become connected components via union-find
//!   ([`cluster::cluster_edges`]).
//!
//! # The bit-parity invariant
//!
//! [`frame_hash::dhash_9x8`] and the sampling geometry around it are fixed
//! forever: their output is persisted by consumers, so a changed bit would
//! make stored hashes incomparable. See [`frame_hash`]. Everything else
//! (windows, thresholds, motion bits, overlap floors) is tunable through
//! [`config::DedupParams`].
//!
//! # The motion gate, and reproducing plain averaging
//!
//! The default scoring is motion-gated. Setting
//! [`config::DedupParams::motion_bits`] to `0` disables the gate and scores the
//! plain average over all overlapping frames — the two regimes share one code
//! path, so a consumer can pick either without a separate implementation.
//!
//! # Features
//!
//! All optional; the default build is pure math with zero dependencies.
//!
//! - `serde`: derive `Serialize`/`Deserialize` on the public result and config
//!   types.
//! - `file-hash`: a SHA-256 exact-file digest helper ([`file_hash`]).
//! - `ffmpeg`: frame and audio-fingerprint extraction via `ffmpeg`/`fpcalc`
//!   ([`extract`]).
//! - `image`: decode an image file/bytes to grayscale for the 256-bit path.
//!
//! The boundary is plain slices and `Vec`s throughout, so language bindings
//! (e.g. PyO3) over the primitives stay thin.

pub mod align;
pub mod audio;
pub mod cluster;
pub mod config;
pub mod frame_hash;
pub mod image_hash;

#[cfg(feature = "file-hash")]
pub mod file_hash;

#[cfg(feature = "ffmpeg")]
pub mod extract;

// Re-export the primary entry points at the crate root for ergonomics.
pub use align::{
    Alignment, DupeEdge, OverlapKind, PairScore, best_alignment, find_candidates, score_visual,
    score_visual_segments,
};
pub use audio::{
    AudioDupeEdge, best_audio_alignment, find_audio_candidates, hamming32, score_audio,
};
pub use cluster::cluster_edges;
pub use config::DedupParams;
pub use frame_hash::{SAMPLE_BYTES, SAMPLE_FPS, SAMPLE_H, SAMPLE_W, dhash_9x8, hamming64};
pub use image_hash::{HammingIndex, ImageHash, ImageHashError, find_duplicates};

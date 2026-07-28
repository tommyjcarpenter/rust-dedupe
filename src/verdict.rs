//! Turning a scored pair into a duplicate/not-duplicate answer.
//!
//! [`crate::align`] and [`crate::audio`] measure how alike two clips are;
//! this module decides what that measurement means. The two are separate on
//! purpose — a consumer can re-run the policy over stored scores without
//! re-hashing anything, and can tune the thresholds without touching the
//! measurement.
//!
//! # The rule
//!
//! Visual evidence is admitted alone only when it is overwhelming. A coarse
//! 64-bit frame hash cannot reliably separate visually homogeneous footage: two
//! unrelated moments from the same scene, performer or set can score in the
//! single digits. Anything short of near-identical therefore needs a second,
//! independent signal before it counts as a duplicate.
//!
//! ```text
//! visual_avg <= near_identical_visual_bits        -> duplicate (vision alone)
//! visual_avg <= threshold_bits, and audio agrees  -> duplicate (corroborated)
//! anything else                                   -> not a duplicate
//! ```
//!
//! Audio never admits a pair by itself. Clips can share a soundtrack — the same
//! backing track, the same room tone — while showing entirely different footage,
//! so treating an audio match as sufficient produces false positives in bulk.
//! Audio only confirms what the visual signal already put in range.
//!
//! Near-identical is also the only route for a silent clip, which by definition
//! has no audio to corroborate with.

use crate::align::PairScore;
use crate::config::DedupParams;

/// Why a pair qualified as a duplicate.
///
/// Consumers that report or count duplicates by category can match on this;
/// consumers that only care whether a pair matched can treat any [`Some`] from
/// [`classify_pair`] as a duplicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DupeVerdict {
    /// The frames are near-identical, so the visual signal stands on its own.
    /// This is genuinely repeated content — a re-encode, a re-upload, a
    /// concatenated copy — and is the only verdict available to silent clips.
    NearIdentical,
    /// The visual match was within the outer threshold but not near-identical,
    /// and the audio agreed the two clips are the same footage.
    AudioCorroborated,
}

/// Decide whether a scored pair is a duplicate, and on what evidence.
///
/// Returns `None` when the pair is not a duplicate: either the visual distance
/// is past [`DedupParams::threshold_bits`], or it is inside that band but the
/// audio is missing, too distant, or was never measured.
///
/// The relevant parameters are [`DedupParams::near_identical_visual_bits`]
/// (below which vision alone suffices), [`DedupParams::threshold_bits`] (the
/// outer ceiling), and [`DedupParams::audio_threshold_bits`] (how close the
/// audio must be to corroborate). Both visual bounds are inclusive.
///
/// Setting `near_identical_visual_bits` equal to `threshold_bits` disables the
/// corroborated tier, admitting only near-identical pairs — a useful escape
/// hatch if audio fingerprints are unavailable or unreliable for a corpus.
/// Setting it to `0.0` has the opposite effect, requiring audio for everything
/// except a bit-exact visual match.
///
/// `score.audio` is populated by the caller (see [`crate::audio::score_audio`]),
/// which lets it be computed lazily — only pairs that land in the corroboration
/// band ever need it.
///
/// # Examples
///
/// A near-identical pair is a duplicate with no audio at all:
///
/// ```
/// use perceptual_dedupe::{DedupParams, DupeVerdict, PairScore, classify_pair};
///
/// let params = DedupParams::default();
/// let score = PairScore {
///     visual_avg: 1.0,
///     visual_overlap: 60,
///     audio: None,
///     visual_seg_a: 0,
///     visual_seg_b: 0,
/// };
/// assert_eq!(classify_pair(&score, &params), Some(DupeVerdict::NearIdentical));
/// ```
///
/// A looser visual match needs the audio to agree:
///
/// ```
/// use perceptual_dedupe::{DedupParams, DupeVerdict, PairScore, classify_pair};
///
/// let params = DedupParams::default();
/// let borderline = PairScore {
///     visual_avg: 8.0,
///     visual_overlap: 60,
///     audio: None,
///     visual_seg_a: 0,
///     visual_seg_b: 0,
/// };
/// assert_eq!(classify_pair(&borderline, &params), None);
///
/// let corroborated = PairScore {
///     audio: Some((1.0, 60)),
///     ..borderline
/// };
/// assert_eq!(
///     classify_pair(&corroborated, &params),
///     Some(DupeVerdict::AudioCorroborated)
/// );
/// ```
pub fn classify_pair(score: &PairScore, params: &DedupParams) -> Option<DupeVerdict> {
    if score.visual_avg <= params.near_identical_visual_bits {
        return Some(DupeVerdict::NearIdentical);
    }
    if score.visual_avg > params.threshold_bits {
        return None;
    }
    match score.audio {
        Some((audio_avg, _)) if audio_avg <= params.audio_threshold_bits => {
            Some(DupeVerdict::AudioCorroborated)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct from the defaults so a test can't pass by coincidence.
    fn params() -> DedupParams {
        DedupParams {
            near_identical_visual_bits: 3.0,
            threshold_bits: 10.0,
            audio_threshold_bits: 2.0,
            ..DedupParams::default()
        }
    }

    fn score(visual_avg: f32, audio: Option<(f32, usize)>) -> PairScore {
        PairScore {
            visual_avg,
            visual_overlap: 60,
            audio,
            visual_seg_a: 0,
            visual_seg_b: 0,
        }
    }

    #[test]
    fn near_identical_admits_without_audio() {
        assert_eq!(
            classify_pair(&score(0.0, None), &params()),
            Some(DupeVerdict::NearIdentical)
        );
    }

    #[test]
    fn near_identical_bound_is_inclusive() {
        assert_eq!(
            classify_pair(&score(3.0, None), &params()),
            Some(DupeVerdict::NearIdentical)
        );
    }

    #[test]
    fn near_identical_ignores_disagreeing_audio() {
        // Vision this close is conclusive on its own; a silent clip and a clip
        // whose audio was re-encoded past the audio threshold must both still
        // register as the repeated content they are.
        assert_eq!(
            classify_pair(&score(1.0, Some((31.0, 60))), &params()),
            Some(DupeVerdict::NearIdentical)
        );
    }

    #[test]
    fn borderline_needs_audio() {
        assert_eq!(classify_pair(&score(8.0, None), &params()), None);
        assert_eq!(
            classify_pair(&score(8.0, Some((1.0, 60))), &params()),
            Some(DupeVerdict::AudioCorroborated)
        );
    }

    #[test]
    fn borderline_rejects_distant_audio() {
        assert_eq!(
            classify_pair(&score(8.0, Some((2.01, 60))), &params()),
            None
        );
    }

    #[test]
    fn audio_bound_is_inclusive() {
        assert_eq!(
            classify_pair(&score(8.0, Some((2.0, 60))), &params()),
            Some(DupeVerdict::AudioCorroborated)
        );
    }

    #[test]
    fn outer_ceiling_is_inclusive() {
        assert_eq!(
            classify_pair(&score(10.0, Some((0.0, 60))), &params()),
            Some(DupeVerdict::AudioCorroborated)
        );
    }

    #[test]
    fn past_the_ceiling_nothing_admits() {
        // Perfect audio must not rescue a pair the visual signal rejected —
        // clips sharing a soundtrack are usually unrelated footage.
        assert_eq!(
            classify_pair(&score(10.01, Some((0.0, 60))), &params()),
            None
        );
        assert_eq!(
            classify_pair(&score(64.0, Some((0.0, 60))), &params()),
            None
        );
    }

    #[test]
    fn equal_thresholds_disable_the_corroborated_tier() {
        let p = DedupParams {
            near_identical_visual_bits: 10.0,
            threshold_bits: 10.0,
            ..params()
        };
        assert_eq!(
            classify_pair(&score(10.0, None), &p),
            Some(DupeVerdict::NearIdentical)
        );
        assert_eq!(classify_pair(&score(10.01, Some((0.0, 60))), &p), None);
    }

    #[test]
    fn zero_near_identical_requires_audio_for_everything_inexact() {
        let p = DedupParams {
            near_identical_visual_bits: 0.0,
            ..params()
        };
        assert_eq!(
            classify_pair(&score(0.0, None), &p),
            Some(DupeVerdict::NearIdentical),
            "a bit-exact visual match still needs nothing"
        );
        assert_eq!(classify_pair(&score(0.5, None), &p), None);
        assert_eq!(
            classify_pair(&score(0.5, Some((1.0, 60))), &p),
            Some(DupeVerdict::AudioCorroborated)
        );
    }
}

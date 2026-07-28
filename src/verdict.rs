//! Turning a scored pair into a duplicate/not-duplicate answer.
//!
//! [`crate::align`] and [`crate::audio`] measure how alike two clips are; this module decides what that
//! measurement means. The two are separate on purpose — a consumer can re-run the policy over stored scores
//! without re-hashing anything, and can tune the thresholds without touching the measurement.
//!
//! # The rule
//!
//! Visual evidence is admitted alone only when it is overwhelming. A coarse 64-bit frame hash cannot reliably
//! separate visually homogeneous footage: two unrelated moments from the same scene, performer or set can score
//! in the single digits. Anything short of near-identical therefore needs a second, independent signal before it
//! counts as a duplicate.
//!
//! ```text
//! visual_avg <= near_identical_visual_bits        -> duplicate (vision alone)
//! visual_avg <= threshold_bits, and audio agrees  -> duplicate (corroborated)
//! anything else                                   -> not a duplicate
//! ```
//!
//! Audio never admits a pair by itself. Clips can share a soundtrack — the same backing track, the same room
//! tone — while showing entirely different footage, so treating an audio match as sufficient produces false
//! positives in bulk. Audio only confirms what the visual signal already put in range.
//!
//! Near-identical is also the only route for a silent clip, which by definition has no audio to corroborate with.
//!
//! # Choosing an entry point
//!
//! [`classify`] takes the two distances the rule actually depends on. [`classify_pair`] is the same decision for
//! a caller that already holds a [`PairScore`]. [`needs_audio_corroboration`] answers the question that comes
//! first in a lazy pipeline: whether this pair's audio is worth scoring at all.
//!
//! Overlap floors are the scorer's job, not the verdict's. [`crate::audio::score_audio`] and
//! [`crate::align::score_visual`] already return `None` below their floors, so a distance reaching this module
//! has cleared them. That is why the decision takes averages alone, and why a caller that scored audio through
//! some other path can still use it.

use crate::align::PairScore;
use crate::config::DedupParams;

/// Why a pair qualified as a duplicate.
///
/// Consumers that report or count duplicates by category can match on this; consumers that only care whether a
/// pair matched can treat any [`Some`] from [`classify`] as a duplicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DupeVerdict {
    /// The frames are near-identical, so the visual signal stands on its own. This is genuinely repeated content
    /// — a re-encode, a re-upload, a concatenated copy — and is the only verdict available to silent clips.
    NearIdentical,
    /// The visual match was within the outer threshold but not near-identical, and the audio agreed the two
    /// clips are the same footage.
    AudioCorroborated,
}

/// Is this pair's audio worth scoring? True when the visual distance lands between the near-identical floor and
/// the outer ceiling, so vision alone settles nothing and only audio can decide.
///
/// Pairs below the floor already qualify and pairs above the ceiling never will, so neither needs audio. A
/// non-finite distance is never in the band.
///
/// This is the predicate for a lazy pipeline that scores audio only where it changes an answer. It shares its
/// bounds with [`classify`], so the two cannot disagree about which pairs audio decides — a band narrower than
/// the verdict would starve pairs of the audio they needed and silently drop matches.
///
/// ```
/// use perceptual_dedupe::{DedupParams, needs_audio_corroboration};
///
/// let params = DedupParams::default(); // near-identical 3.0, ceiling 10.0
/// assert!(!needs_audio_corroboration(1.0, &params)); // already qualifies
/// assert!(needs_audio_corroboration(8.0, &params)); // audio decides
/// assert!(!needs_audio_corroboration(20.0, &params)); // never qualifies
/// ```
pub fn needs_audio_corroboration(visual_avg: f32, params: &DedupParams) -> bool {
    /* Mirrors `classify`'s structure, and for the same reason: phrasing both bounds as "within" fails closed on
    a non-finite one. A NaN floor makes `visual_avg > floor` false, which would report that audio is not worth
    scoring for a pair whose verdict audio actually decides — starving it and silently dropping the match. */
    let within_ceiling = visual_avg.is_finite() && visual_avg <= params.threshold_bits;
    let within_near_identical = visual_avg <= params.near_identical_visual_bits;
    // A non-finite audio bound can never be cleared, so audio could not change the verdict for any pair and
    // scoring it is wasted work. Answering "no" here is what keeps the band equal to the set the verdict cares
    // about rather than merely a subset of it.
    let audio_can_corroborate = params.audio_threshold_bits.is_finite();
    within_ceiling && !within_near_identical && audio_can_corroborate
}

/// Decide whether a pair is a duplicate, and on what evidence.
///
/// Takes the two distances the rule depends on: the average per-frame visual distance, and the average audio
/// sub-fingerprint distance if one was measured. `None` for `audio_avg` means audio was not scored, which is
/// treated the same as audio that disagrees — a pair needing corroboration and lacking it is not a duplicate.
/// Callers that need to tell those apart should ask [`needs_audio_corroboration`] before scoring.
///
/// Returns `None` when the pair is not a duplicate: either the visual distance is past
/// [`DedupParams::threshold_bits`], or it is inside that band but the audio is missing or too distant.
///
/// The parameters read are [`DedupParams::near_identical_visual_bits`] (below which vision alone suffices),
/// [`DedupParams::threshold_bits`] (the outer ceiling), and [`DedupParams::audio_threshold_bits`] (how close the
/// audio must be to corroborate). All three bounds are inclusive.
///
/// [`DedupParams::threshold_bits`] is the outer bound in every case, so a configuration where
/// `near_identical_visual_bits` exceeds it admits nothing past the ceiling. A non-finite distance is never a
/// duplicate.
///
/// Setting `near_identical_visual_bits` equal to `threshold_bits` disables the corroborated tier, admitting only
/// near-identical pairs — a useful escape hatch if audio fingerprints are unavailable or unreliable for a
/// corpus. Setting it to `0.0` has the opposite effect, requiring audio for everything except a bit-exact visual
/// match.
///
/// ```
/// use perceptual_dedupe::{DedupParams, DupeVerdict, classify};
///
/// let params = DedupParams::default();
///
/// // Near-identical needs no audio at all.
/// assert_eq!(classify(1.0, None, &params), Some(DupeVerdict::NearIdentical));
///
/// // A looser visual match needs the audio to agree.
/// assert_eq!(classify(8.0, None, &params), None);
/// assert_eq!(classify(8.0, Some(1.0), &params), Some(DupeVerdict::AudioCorroborated));
///
/// // Past the ceiling, perfect audio changes nothing.
/// assert_eq!(classify(20.0, Some(0.0), &params), None);
/// ```
pub fn classify(
    visual_avg: f32,
    audio_avg: Option<f32>,
    params: &DedupParams,
) -> Option<DupeVerdict> {
    /* The outer ceiling is enforced first so it always binds. Checking near-identical first would let a
    configuration with `near_identical_visual_bits > threshold_bits` admit a pair the ceiling rejects.

    Phrased as "not within the ceiling" rather than "past the ceiling" so it fails closed on a non-finite bound:
    every comparison against NaN is false, so `visual_avg > NaN` would skip the return and leave the pair to be
    decided by its audio alone, which the rule never permits. The same phrasing rejects a non-finite distance. */
    let within_ceiling = visual_avg.is_finite() && visual_avg <= params.threshold_bits;
    if !within_ceiling {
        return None;
    }
    if visual_avg <= params.near_identical_visual_bits {
        return Some(DupeVerdict::NearIdentical);
    }
    match audio_avg {
        Some(audio) if audio <= params.audio_threshold_bits => Some(DupeVerdict::AudioCorroborated),
        _ => None,
    }
}

/// [`classify`] for a caller holding a [`PairScore`], using its visual distance and the average from its lazily
/// filled audio alignment.
///
/// The overlap counts on the score are not consulted; see the module docs on why floors belong to the scorer.
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
pub fn classify_pair(score: &PairScore, params: &DedupParams) -> Option<DupeVerdict> {
    classify(score.visual_avg, score.audio.map(|(avg, _)| avg), params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::Alignment;

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
            classify(0.0, None, &params()),
            Some(DupeVerdict::NearIdentical)
        );
    }

    #[test]
    fn near_identical_bound_is_inclusive() {
        assert_eq!(
            classify(3.0, None, &params()),
            Some(DupeVerdict::NearIdentical)
        );
    }

    #[test]
    fn near_identical_ignores_disagreeing_audio() {
        /* Vision this close is conclusive on its own; a silent clip and a clip whose audio was re-encoded past
        the audio threshold must both still register as the repeated content they are. */
        assert_eq!(
            classify(1.0, Some(31.0), &params()),
            Some(DupeVerdict::NearIdentical)
        );
    }

    #[test]
    fn borderline_needs_audio() {
        assert_eq!(classify(8.0, None, &params()), None);
        assert_eq!(
            classify(8.0, Some(1.0), &params()),
            Some(DupeVerdict::AudioCorroborated)
        );
    }

    #[test]
    fn borderline_rejects_distant_audio() {
        assert_eq!(classify(8.0, Some(2.01), &params()), None);
    }

    #[test]
    fn audio_bound_is_inclusive() {
        assert_eq!(
            classify(8.0, Some(2.0), &params()),
            Some(DupeVerdict::AudioCorroborated)
        );
    }

    #[test]
    fn outer_ceiling_is_inclusive() {
        assert_eq!(
            classify(10.0, Some(0.0), &params()),
            Some(DupeVerdict::AudioCorroborated)
        );
    }

    #[test]
    fn past_the_ceiling_nothing_admits() {
        // Perfect audio must not rescue a pair the visual signal rejected — clips
        // sharing a soundtrack are usually unrelated footage.
        assert_eq!(classify(10.01, Some(0.0), &params()), None);
        assert_eq!(classify(64.0, Some(0.0), &params()), None);
    }

    #[test]
    fn ceiling_binds_even_if_near_identical_is_configured_above_it() {
        /* This config does widen the near-identical tier — every in-ceiling pair becomes NearIdentical. What it
        must not do is admit anything past the ceiling, which is the invariant asserted here. */
        let p = DedupParams {
            near_identical_visual_bits: 20.0,
            threshold_bits: 10.0,
            ..params()
        };
        assert_eq!(classify(10.0, None, &p), Some(DupeVerdict::NearIdentical));
        assert_eq!(classify(10.01, None, &p), None);
    }

    #[test]
    fn the_no_match_sentinel_is_not_a_duplicate() {
        /* `best_alignment` reports "no qualifying alignment" as f32::MAX, which is finite and so reaches the
        ceiling check rather than the non-finite guard. It is the one extreme value the crate itself produces. */
        assert_eq!(
            classify(Alignment::NO_MATCH.avg_bits, Some(0.0), &params()),
            None
        );
    }

    #[test]
    fn a_non_finite_ceiling_still_binds() {
        /* A NaN ceiling makes every comparison against it false. Phrased as "past the ceiling" the pair would
        fall through to the audio arm and be admitted on audio alone, contradicting the rule. */
        let p = DedupParams {
            threshold_bits: f32::NAN,
            ..params()
        };
        assert_eq!(classify(8.0, Some(0.0), &p), None);
        assert!(!needs_audio_corroboration(8.0, &p));
    }

    #[test]
    fn non_finite_visual_distance_is_never_a_duplicate() {
        // NaN fails every comparison, so without an explicit guard it would slip
        // past both bounds and be decided by the audio alone.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(classify(bad, Some(0.0), &params()), None);
            assert_eq!(classify(bad, None, &params()), None);
        }
    }

    #[test]
    fn non_finite_audio_distance_does_not_corroborate() {
        assert_eq!(classify(8.0, Some(f32::NAN), &params()), None);
    }

    #[test]
    fn equal_thresholds_disable_the_corroborated_tier() {
        let p = DedupParams {
            near_identical_visual_bits: 10.0,
            threshold_bits: 10.0,
            ..params()
        };
        assert_eq!(classify(10.0, None, &p), Some(DupeVerdict::NearIdentical));
        assert_eq!(classify(10.01, Some(0.0), &p), None);
    }

    #[test]
    fn zero_near_identical_requires_audio_for_everything_inexact() {
        let p = DedupParams {
            near_identical_visual_bits: 0.0,
            ..params()
        };
        let exact = classify(0.0, None, &p);
        assert_eq!(
            exact,
            Some(DupeVerdict::NearIdentical),
            "a bit-exact match still needs nothing"
        );
        assert_eq!(classify(0.5, None, &p), None);
        assert_eq!(
            classify(0.5, Some(1.0), &p),
            Some(DupeVerdict::AudioCorroborated)
        );
    }

    #[test]
    fn classify_pair_matches_the_scalar_form() {
        /* The wrapper must stay a pure forwarding of the two distances, or the two entry points drift into
        different rules. */
        let p = params();
        for visual in [0.0f32, 3.0, 8.0, 10.0, 10.01, f32::NAN] {
            for audio in [None, Some(1.0f32), Some(31.0)] {
                assert_eq!(
                    classify_pair(&score(visual, audio.map(|a| (a, 60))), &p),
                    classify(visual, audio, &p),
                    "visual={visual} audio={audio:?}"
                );
            }
        }
    }

    #[test]
    fn classify_pair_ignores_overlap_counts() {
        /* Overlap floors are the scorer's job; a caller carrying only averages must get the same answer as one
        carrying full alignments. */
        let mut zero_overlap = score(8.0, Some((1.0, 0)));
        zero_overlap.visual_overlap = 0;
        assert_eq!(
            classify_pair(&zero_overlap, &params()),
            Some(DupeVerdict::AudioCorroborated)
        );
    }

    #[test]
    fn the_band_is_exactly_where_audio_changes_the_answer() {
        /* The predicate and the verdict must agree on which pairs audio decides, including under a malformed
        config. A band narrower than the verdict would starve pairs of the audio they needed, and the matches
        would simply stop appearing. The NaN rows are what make "these two share their bounds" true rather than
        true-for-realistic-inputs; the audio one already passes, and holding it here stops a future edit to the
        audio arm quietly breaking it. */
        let configs = [
            params(),
            DedupParams {
                near_identical_visual_bits: f32::NAN,
                ..params()
            },
            DedupParams {
                threshold_bits: f32::NAN,
                ..params()
            },
            DedupParams {
                audio_threshold_bits: f32::NAN,
                ..params()
            },
        ];
        for p in configs {
            for visual in [0.0f32, 3.0, 3.01, 8.0, 10.0, 10.01, 64.0, f32::NAN] {
                let decided_by_audio =
                    classify(visual, None, &p) != classify(visual, Some(0.0), &p);
                assert_eq!(
                    needs_audio_corroboration(visual, &p),
                    decided_by_audio,
                    "visual={visual} params={p:?}"
                );
            }
        }
    }

    #[test]
    fn band_excludes_non_finite_distances() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!needs_audio_corroboration(bad, &params()));
        }
    }
}

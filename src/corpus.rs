//! A queryable corpus index over many items' `u64` frame-hash sequences.
//!
//! [`find_candidates`](crate::find_candidates) answers "which pairs in THIS set
//! are dupes" in one all-pairs pass. This module answers a different question:
//! "which items in a large, persistent corpus could match THIS one query
//! sequence" — without rescanning the whole corpus per query. That's what a
//! historical / "seen before" corpus needs: build the index once, query it per
//! incoming clip.
//!
//! It's a pigeonhole band index, the `u64`-frame analogue of
//! [`HammingIndex`](crate::HammingIndex) (which serves the 256-bit `ImageHash`
//! path): each 64-bit frame hash is split into `bands` equal-width bands, and an
//! item is filed in the bucket for each band's value. A query unions the buckets
//! its own frames' bands fall in.
//!
//! The `bands` count is a per-application knob ([`FrameCorpusIndex::with_bands`])
//! trading recall against selectivity — see the type docs. Whatever the setting,
//! two frames at most `bands - 1` bits of Hamming distance apart are *guaranteed*
//! to share a band (pigeonhole), so a near-identical frame is never missed;
//! farther-apart frames may still collide, just without the guarantee. Because a
//! real duplicate shares MANY near-identical frames, indexing every frame catches
//! it via its closest ones. This is purely a **candidate prefilter**: the caller
//! must verify every candidate with
//! [`score_visual_segments`](crate::score_visual_segments) (or
//! [`score_visual`](crate::score_visual)) — a genuinely borderline match with no
//! near-identical frame can be missed, which is acceptable for an advisory "seen
//! before" signal but not for a hard dedup guarantee.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Default band count for [`FrameCorpusIndex::new`]. Eight 8-bit bands favour
/// recall — two frames within 7 bits of Hamming distance still share a band — so
/// same-content-different-encode duplicates (frames uniformly a few bits off) are
/// still caught. Use [`FrameCorpusIndex::with_bands`] to trade recall for
/// selectivity on a very large corpus (see that method).
pub const DEFAULT_BANDS: usize = 8;

#[inline]
fn band(hash: u64, i: usize, width: u32) -> u16 {
    let mask: u64 = (1u64 << width) - 1;
    ((hash >> (i as u32 * width)) & mask) as u16
}

/// Corpus index mapping each band value to the ids of items that have a frame
/// with that band. Build with [`new`](Self::new) or [`with_bands`](Self::with_bands),
/// populate with [`add`](Self::add), then [`query`](Self::query) per incoming
/// sequence. Append-only; to forget an item, rebuild the index (corpus prunes are
/// rare relative to queries).
///
/// The band count is the single tuning knob (recall vs selectivity):
/// - **Fewer bands** (wider, e.g. 4 × 16-bit) → tiny buckets → tiny candidate
///   sets, best on a huge corpus — but only near-identical frames (≤ `bands - 1`
///   bits, pigeonhole) are guaranteed to land together, so a uniformly-offset
///   re-encode can be missed.
/// - **More bands** (narrower, e.g. 8 × 8-bit) → catches farther-apart frames
///   (better recall) at the cost of larger buckets / candidate sets.
///
/// Either way it is only a candidate prefilter: the caller confirms each
/// candidate with [`score_visual_segments`](crate::score_visual_segments).
///
/// ```
/// use perceptual_dedupe::FrameCorpusIndex;
///
/// let mut index = FrameCorpusIndex::new();
/// index.add("clip_a", [0x1122_3344_5566_7788_u64, 0x99AA_BBCC_DDEE_FF00]);
/// index.add("clip_b", [0x0000_0000_0000_0001_u64]);
///
/// // A query sharing a frame with clip_a comes back as a candidate to verify.
/// let candidates = index.query([0x1122_3344_5566_7788_u64]);
/// assert!(candidates.contains(&"clip_a"));
/// assert!(!candidates.contains(&"clip_b"));
/// ```
#[derive(Debug, Clone)]
pub struct FrameCorpusIndex<Id> {
    tables: Vec<HashMap<u16, Vec<Id>>>,
    band_width: u32,
}

// Construction / inspection need no bounds on `Id` (only `add`/`query` do), so a
// `FrameCorpusIndex<Id>` is `Default`-constructible before `Id` is constrained.
impl<Id> Default for FrameCorpusIndex<Id> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Id> FrameCorpusIndex<Id> {
    /// Build an empty index with [`DEFAULT_BANDS`] bands.
    pub fn new() -> Self {
        Self::with_bands(DEFAULT_BANDS)
    }

    /// Build an empty index that splits each 64-bit frame hash into `bands`
    /// equal bands. `bands` must divide 64 and leave each band at most 16 bits
    /// wide (so `bands` in `{4, 8, 16, 32, 64}`); other values panic. See the
    /// type docs for the recall-vs-selectivity trade-off.
    pub fn with_bands(bands: usize) -> Self {
        assert!(
            bands != 0 && 64 % bands == 0 && 64 / bands <= 16,
            "bands must divide 64 with width <= 16 bits (one of 4, 8, 16, 32, 64), got {bands}"
        );
        Self {
            tables: (0..bands).map(|_| HashMap::new()).collect(),
            band_width: (64 / bands) as u32,
        }
    }

    /// True when no item has been indexed yet.
    pub fn is_empty(&self) -> bool {
        self.tables.iter().all(HashMap::is_empty)
    }
}

impl<Id: Copy + Eq + Hash> FrameCorpusIndex<Id> {
    /// Index `frames` under `id`. Within this call `id` lands at most once per
    /// (band, value) bucket, so an item with many similar frames doesn't bloat
    /// the buckets. Adding the same `id` across multiple calls is NOT deduplicated
    /// (it can then appear more than once in a bucket, though [`query`](Self::query)
    /// still returns it once) — rebuild the index to replace an item.
    pub fn add(&mut self, id: Id, frames: impl IntoIterator<Item = u64>) {
        let mut seen: HashSet<(usize, u16)> = HashSet::new();
        for f in frames {
            for (i, table) in self.tables.iter_mut().enumerate() {
                let v = band(f, i, self.band_width);
                if seen.insert((i, v)) {
                    table.entry(v).or_default().push(id);
                }
            }
        }
    }

    /// Ids that share at least one frame band with `frames` — the candidate set
    /// to verify with a full score. Each distinct query band is scanned once.
    pub fn query(&self, frames: impl IntoIterator<Item = u64>) -> HashSet<Id> {
        let mut out: HashSet<Id> = HashSet::new();
        let mut scanned: HashSet<(usize, u16)> = HashSet::new();
        for f in frames {
            for (i, table) in self.tables.iter().enumerate() {
                let v = band(f, i, self.band_width);
                if scanned.insert((i, v))
                    && let Some(bucket) = table.get(&v)
                {
                    out.extend(bucket.iter().copied());
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_frame_is_a_candidate() {
        let mut ix = FrameCorpusIndex::new();
        ix.add(1u32, [0xDEAD_BEEF_CAFE_F00Du64]);
        ix.add(2u32, [0x0123_4567_89AB_CDEFu64]);
        let c = ix.query([0xDEAD_BEEF_CAFE_F00Du64]);
        assert!(c.contains(&1));
        assert!(!c.contains(&2), "unrelated hash shares no band here");
    }

    #[test]
    fn near_identical_frame_shares_a_band() {
        // Flip 2 bits (< the band count): pigeonhole guarantees an identical band.
        let base = 0xAAAA_AAAA_AAAA_AAAAu64;
        let near = base ^ 0b11; // two bits in the lowest band
        let mut ix = FrameCorpusIndex::new();
        ix.add(7u32, [base]);
        assert!(ix.query([near]).contains(&7));
    }

    #[test]
    fn with_bands_builds_a_working_index() {
        // Strict setting (4 x 16-bit bands) still finds an identical frame.
        let mut ix = FrameCorpusIndex::with_bands(4);
        ix.add(9u32, [0xDEAD_BEEF_CAFE_F00Du64]);
        assert!(ix.query([0xDEAD_BEEF_CAFE_F00Du64]).contains(&9));
    }

    #[test]
    #[should_panic]
    fn with_bands_rejects_a_non_divisor_of_64() {
        let _: FrameCorpusIndex<u32> = FrameCorpusIndex::with_bands(3);
    }

    #[test]
    #[should_panic]
    fn with_bands_rejects_bands_wider_than_16_bits() {
        let _: FrameCorpusIndex<u32> = FrameCorpusIndex::with_bands(2);
    }

    #[test]
    fn unrelated_frames_do_not_collide() {
        let mut ix = FrameCorpusIndex::new();
        ix.add(1u32, [0x0000_0000_0000_0000u64]);
        // Every band differs from the query, so no shared bucket.
        assert!(ix.query([0x1111_2222_3333_4444u64]).is_empty());
    }

    #[test]
    fn candidate_via_any_shared_frame_in_the_sequence() {
        // The query's second frame matches item 5's first frame → still a candidate.
        let mut ix = FrameCorpusIndex::new();
        ix.add(5u32, [0xFFFF_0000_FFFF_0000u64, 0x1234_1234_1234_1234u64]);
        let c = ix.query([0x9999_9999_9999_9999u64, 0xFFFF_0000_FFFF_0000u64]);
        assert!(c.contains(&5));
    }

    #[test]
    fn empty_index_returns_no_candidates() {
        let ix: FrameCorpusIndex<u32> = FrameCorpusIndex::new();
        assert!(ix.is_empty());
        assert!(ix.query([1u64, 2, 3]).is_empty());
    }
}

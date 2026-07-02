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
//! path): each 64-bit frame hash is split into [`BANDS`] 16-bit bands, and an
//! item is filed in the bucket for each band's value. A query unions the buckets
//! its own frames' bands fall in.
//!
//! Two frames within `BANDS - 1` bits of Hamming distance are *guaranteed* to
//! share a band (pigeonhole), so a near-identical frame is never missed; frames
//! farther apart may still collide, just without the guarantee. Because a real
//! duplicate shares MANY near-identical frames, indexing every frame catches it
//! via its closest ones even though the per-frame guarantee is small. This is
//! purely a **candidate prefilter**: it errs toward recall, and the caller must
//! verify every candidate with [`score_visual_segments`](crate::score_visual_segments)
//! (or [`score_visual`](crate::score_visual)) — a genuinely borderline match with
//! no near-identical frame can be missed, which is acceptable for an advisory
//! "seen before" signal but not for a hard dedup guarantee.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Number of 16-bit bands a 64-bit frame hash is split into. 16-bit bands keep
/// buckets small (65536 values per band position) so selectivity holds on a
/// large corpus; the pigeonhole recall guarantee covers per-frame Hamming
/// distance below this count.
pub const BANDS: usize = 4;

#[inline]
fn band(hash: u64, i: usize) -> u16 {
    (hash >> (i * 16)) as u16
}

/// Corpus index mapping each band value to the ids of items that have a frame
/// with that band. Build with [`FrameCorpusIndex::new`] + [`add`](Self::add),
/// then [`query`](Self::query) per incoming sequence. Append-only; to forget an
/// item, rebuild the index (corpus prunes are rare relative to queries).
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
pub struct FrameCorpusIndex<Id> {
    tables: [HashMap<u16, Vec<Id>>; BANDS],
}

impl<Id: Copy + Eq + Hash> Default for FrameCorpusIndex<Id> {
    fn default() -> Self {
        Self {
            tables: std::array::from_fn(|_| HashMap::new()),
        }
    }
}

impl<Id: Copy + Eq + Hash> FrameCorpusIndex<Id> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index `frames` under `id`. An id lands at most once per (band, value)
    /// bucket, so an item with many similar frames doesn't bloat the buckets.
    pub fn add(&mut self, id: Id, frames: impl IntoIterator<Item = u64>) {
        let mut seen: HashSet<(usize, u16)> = HashSet::new();
        for f in frames {
            for i in 0..BANDS {
                let v = band(f, i);
                if seen.insert((i, v)) {
                    self.tables[i].entry(v).or_default().push(id);
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
            for i in 0..BANDS {
                let v = band(f, i);
                if scanned.insert((i, v))
                    && let Some(bucket) = self.tables[i].get(&v)
                {
                    out.extend(bucket.iter().copied());
                }
            }
        }
        out
    }

    /// True when no item has been indexed yet.
    pub fn is_empty(&self) -> bool {
        self.tables.iter().all(HashMap::is_empty)
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
        // Flip 2 bits (< BANDS): pigeonhole guarantees at least one identical band.
        let base = 0xAAAA_AAAA_AAAA_AAAAu64;
        let near = base ^ 0b11; // two bits in the lowest band
        let mut ix = FrameCorpusIndex::new();
        ix.add(7u32, [base]);
        assert!(ix.query([near]).contains(&7));
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

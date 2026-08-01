//! A queryable corpus index over many items' hash sequences.
//!
//! [`find_candidates`](crate::find_candidates) answers "which pairs in THIS set
//! are dupes" in one all-pairs pass. This module answers a different question:
//! "which items in a large, persistent corpus could match THIS one query
//! sequence" — without rescanning the whole corpus per query. That's what a
//! historical / "seen before" corpus needs: build the index once, query it per
//! incoming clip.
//!
//! It's a pigeonhole band index, the analogue of
//! [`HammingIndex`](crate::HammingIndex) (which serves the 256-bit `ImageHash`
//! path): each hash is split into `bands` equal-width bands, and an item is filed
//! in the bucket for each band's value. A query unions the buckets its own
//! elements' bands fall in. Works over `u64` frame hashes and, as
//! [`AudioCorpusIndex`], over `u32` audio sub-fingerprints.
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

/// Default band count for [`FrameCorpusIndex::new`]. Eight bands favour recall: two hashes
/// within 7 bits of Hamming distance still share one, so same-content-different-encode
/// duplicates (uniformly a few bits off) are still caught.
///
/// The guarantee is `bands - 1` bits whatever the hash width, so it holds equally for `u64`
/// frames and `u32` sub-fingerprints — only the band WIDTH differs, 8 bits on the former and 4
/// on the latter. Narrower bands mean larger buckets, so the same count is somewhat less
/// selective on the shorter hash.
///
/// Use [`FrameCorpusIndex::with_bands`] to trade recall for selectivity on a large corpus.
pub const DEFAULT_BANDS: usize = 8;

mod sealed {
    pub trait Sealed {}
    impl Sealed for u64 {}
    impl Sealed for u32 {}
}

/// A fixed-width hash the index can split into bands. Implemented for `u64` (frame hashes) and
/// `u32` (audio sub-fingerprints); sealed, because the band arithmetic assumes the width is
/// exactly `BITS` and a third implementation would need the assertions revisited.
pub trait BandedHash: sealed::Sealed + Copy + Eq + Hash {
    /// Width of the hash in bits. Bands must divide it.
    const BITS: usize;
    /// The `i`th band of `width` bits, low-order band first.
    fn band(self, i: usize, width: u32) -> u16;
    /// Hamming distance to another hash of the same width.
    fn hamming(self, other: Self) -> u32;
}

impl BandedHash for u64 {
    const BITS: usize = 64;
    #[inline]
    fn band(self, i: usize, width: u32) -> u16 {
        let mask: u64 = (1u64 << width) - 1;
        ((self >> (i as u32 * width)) & mask) as u16
    }
    #[inline]
    fn hamming(self, other: Self) -> u32 {
        (self ^ other).count_ones()
    }
}

impl BandedHash for u32 {
    const BITS: usize = 32;
    #[inline]
    fn band(self, i: usize, width: u32) -> u16 {
        let mask: u32 = (1u32 << width) - 1;
        ((self >> (i as u32 * width)) & mask) as u16
    }
    #[inline]
    fn hamming(self, other: Self) -> u32 {
        (self ^ other).count_ones()
    }
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
pub struct FrameCorpusIndex<Id, H: BandedHash = u64> {
    tables: Vec<HashMap<u16, Vec<(Id, H)>>>,
    band_width: u32,
    /// `fn() -> H` rather than `H`: the index owns no hashes, it only reads their bands.
    hash: std::marker::PhantomData<fn() -> H>,
}

/// [`FrameCorpusIndex`] over audio sub-fingerprints.
///
/// The audio side is usually the one worth indexing. A frame hash cannot see through a reframe,
/// so on re-encoded content no two frames are near-identical and a visual index surfaces nothing;
/// audio reads near-zero on shared content, so its sub-fingerprints do land in the same bands.
pub type AudioCorpusIndex<Id> = FrameCorpusIndex<Id, u32>;

// Construction / inspection need no bounds on `Id` (only `add`/`query` do), so a
// `FrameCorpusIndex<Id>` is `Default`-constructible before `Id` is constrained.
impl<Id, H: BandedHash> Default for FrameCorpusIndex<Id, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Id, H: BandedHash> FrameCorpusIndex<Id, H> {
    /// Build an empty index with [`DEFAULT_BANDS`] bands.
    pub fn new() -> Self {
        Self::with_bands(DEFAULT_BANDS)
    }

    /// Build an empty index that splits each hash into `bands` equal bands. `bands` must divide
    /// the hash width and leave each band at most 16 bits wide; other values panic. See the type
    /// docs for the recall-vs-selectivity trade-off.
    pub fn with_bands(bands: usize) -> Self {
        let bits = H::BITS;
        assert!(
            bands != 0 && bits % bands == 0 && bits / bands <= 16,
            "bands must divide {bits} with width <= 16 bits, got {bands}"
        );
        Self {
            tables: (0..bands).map(|_| HashMap::new()).collect(),
            band_width: (bits / bands) as u32,
            hash: std::marker::PhantomData,
        }
    }

    /// True when no item has been indexed yet.
    pub fn is_empty(&self) -> bool {
        self.tables.iter().all(HashMap::is_empty)
    }
}

impl<Id: Copy + Eq + Hash, H: BandedHash> FrameCorpusIndex<Id, H> {
    /// Index `frames` under `id`. Within this call `id` lands at most once per
    /// (band, value) bucket, so an item with many similar frames doesn't bloat
    /// the buckets. Adding the same `id` across multiple calls is NOT deduplicated
    /// (it can then appear more than once in a bucket, though [`query`](Self::query)
    /// still returns it once) — rebuild the index to replace an item.
    pub fn add(&mut self, id: Id, frames: impl IntoIterator<Item = H>) {
        let mut seen: HashSet<(usize, u16, H)> = HashSet::new();
        for f in frames {
            for (i, table) in self.tables.iter_mut().enumerate() {
                let v = f.band(i, self.band_width);
                /* Deduped on the HASH as well as the bucket, not just the bucket. Keeping one
                representative per bucket would be enough to answer "does anything collide", but
                `query_votes` measures distance against what is stored, so a distinct hash that
                happened to share a bucket with an earlier one has to survive or it can never be
                voted for. Exact repeats — a static run, a silent stretch — still collapse. */
                if seen.insert((i, v, f)) {
                    table.entry(v).or_default().push((id, f));
                }
            }
        }
    }

    /// Ids that share at least one frame band with `frames` — the candidate set
    /// to verify with a full score. Each distinct query band is scanned once.
    pub fn query(&self, frames: impl IntoIterator<Item = H>) -> HashSet<Id> {
        let mut out: HashSet<Id> = HashSet::new();
        let mut scanned: HashSet<(usize, u16)> = HashSet::new();
        for f in frames {
            for (i, table) in self.tables.iter().enumerate() {
                let v = f.band(i, self.band_width);
                if scanned.insert((i, v))
                    && let Some(bucket) = table.get(&v)
                {
                    out.extend(bucket.iter().map(|(id, _)| *id));
                }
            }
        }
        out
    }

    /// Votes per candidate: how many DISTINCT query elements have a match within `max_bits` of
    /// one of that candidate's. The number to threshold on when a bucket collision alone is not
    /// evidence.
    ///
    /// [`query`](Self::query) returns everything sharing any band with any query element. Over a
    /// long query that is nearly the whole corpus: with a few hundred elements across several
    /// bands, chance collisions against any given candidate are all but certain, so the union
    /// selects nothing. Checking the distance is what restores selectivity — near-identity is
    /// rare where collision is not. Within 3 bits of a 32-bit hash lie ~5457 of 2^32 values, so a
    /// chance vote runs about 1.3e-6 per element pair, while genuinely shared content votes once
    /// per shared element.
    ///
    /// `max_bits` must be under the pigeonhole limit (`bands - 1`) or the guarantee is lost and
    /// near matches can be missed: a value farther apart than that need not share any band, so it
    /// is never in a bucket to be checked. Distinct elements, not raw collisions — one query
    /// element colliding on several bands is one vote, so the count is comparable across
    /// candidates however the bands fell.
    pub fn query_votes(
        &self,
        frames: impl IntoIterator<Item = H>,
        max_bits: u32,
    ) -> HashMap<Id, usize> {
        let mut votes: HashMap<Id, usize> = HashMap::new();
        let mut voted: HashSet<(usize, Id)> = HashSet::new();
        for (qi, f) in frames.into_iter().enumerate() {
            for (i, table) in self.tables.iter().enumerate() {
                let v = f.band(i, self.band_width);
                let Some(bucket) = table.get(&v) else {
                    continue;
                };
                for (id, stored) in bucket {
                    if f.hamming(*stored) <= max_bits && voted.insert((qi, *id)) {
                        *votes.entry(*id).or_default() += 1;
                    }
                }
            }
        }
        votes
    }

    /// Candidates with at least `min_votes` elements matching within `max_bits`. See
    /// [`query_votes`](Self::query_votes) for what a vote is and how to choose `max_bits`.
    pub fn query_verified(
        &self,
        frames: impl IntoIterator<Item = H>,
        max_bits: u32,
        min_votes: usize,
    ) -> HashSet<Id> {
        self.query_votes(frames, max_bits)
            .into_iter()
            .filter(|&(_, n)| n >= min_votes)
            .map(|(id, _)| id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /* Audio sub-fingerprints are 32-bit, so the band arithmetic has to key off the hash width
    rather than a baked-in 64. At 4 bands each is 8 bits wide, and the pigeonhole guarantee is
    the same shape: two values within `bands - 1` bits must agree on some band. */
    /* A deterministic pseudo-random sub-fingerprint stream. Real chromaprint output is
    high-entropy, which is the property the selectivity argument rests on. */
    fn fps(seed: u32, n: usize) -> Vec<u32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x
            })
            .collect()
    }

    /* The whole reason `query_votes` exists. A bare bucket union over a few hundred query
    elements returns essentially the entire corpus — chance collisions are certain at that
    length — so it selects nothing and the expensive verification behind it runs on everything.
    Requiring near-identity instead cuts it to the item that actually shares content. */
    #[test]
    fn voting_selects_where_a_bucket_union_does_not() {
        let shared = fps(0xABCD, 60);
        let mut ix: AudioCorpusIndex<u8> = AudioCorpusIndex::with_bands(4);
        for id in 0..40u8 {
            ix.add(id, fps(id as u32 * 7919 + 11, 400));
        }
        // One item genuinely carries the shared run; everything else is unrelated.
        ix.add(
            99u8,
            fps(0x5150, 340)
                .into_iter()
                .chain(shared.clone())
                .collect::<Vec<_>>(),
        );
        let query: Vec<u32> = shared.iter().copied().chain(fps(0x2222, 340)).collect();

        let union = ix.query(query.iter().copied());
        assert!(
            union.len() > 20,
            "a bare union is not selective over a long query: {} of 41",
            union.len()
        );

        let voted = ix.query_verified(query.iter().copied(), 3, 10);
        assert_eq!(
            voted.into_iter().collect::<Vec<_>>(),
            vec![99u8],
            "verification leaves only the item that shares content"
        );
    }

    #[test]
    fn every_shared_element_votes() {
        // A run of N shared elements should produce ~N votes, so a threshold well under the
        // overlap floor is safe rather than a guess.
        let shared = fps(0x1234, 50);
        let mut ix: AudioCorpusIndex<u8> = AudioCorpusIndex::with_bands(4);
        ix.add(1u8, shared.clone());
        let votes = ix.query_votes(shared.iter().copied(), 3);
        assert_eq!(votes.get(&1u8), Some(&50));
    }

    #[test]
    fn a_near_match_inside_the_pigeonhole_limit_still_votes() {
        /* Recall is the property that matters most: a missed overlap fails silently. At 4 bands
        anything within 3 bits must share a band, so it is always in a bucket to be checked. */
        let base = fps(0x777, 40);
        let mut ix: AudioCorpusIndex<u8> = AudioCorpusIndex::with_bands(4);
        ix.add(1u8, base.clone());
        let nudged: Vec<u32> = base.iter().map(|h| h ^ 0b101).collect();
        assert_eq!(ix.query_votes(nudged, 3).get(&1u8), Some(&40));
    }

    #[test]
    fn distance_bounds_the_vote() {
        let base = fps(0x999, 30);
        let mut ix: AudioCorpusIndex<u8> = AudioCorpusIndex::with_bands(4);
        ix.add(1u8, base.clone());
        // Three flipped bits is inside the bound; asking for two excludes them.
        let nudged: Vec<u32> = base.iter().map(|h| h ^ 0b111).collect();
        assert_eq!(ix.query_votes(nudged.clone(), 3).get(&1u8), Some(&30));
        assert_eq!(ix.query_votes(nudged, 2).get(&1u8), None);
    }

    #[test]
    fn an_audio_index_round_trips_a_u32_fingerprint() {
        let mut ix: AudioCorpusIndex<&str> = AudioCorpusIndex::with_bands(4);
        ix.add("clip_a", [0xDEAD_BEEFu32, 0x0123_4567]);
        ix.add("clip_b", [0x0000_0001u32]);
        let c = ix.query([0xDEAD_BEEFu32]);
        assert!(c.contains("clip_a"));
        assert!(!c.contains("clip_b"));
    }

    #[test]
    fn an_audio_index_keeps_the_pigeonhole_guarantee() {
        // 4 bands over 32 bits: anything within 3 bits must collide on at least one band.
        let base = 0xA5A5_A5A5u32;
        for bit_a in 0..32 {
            for bit_b in 0..32 {
                let near = base ^ (1 << bit_a) ^ (1 << bit_b);
                let mut ix: AudioCorpusIndex<u8> = AudioCorpusIndex::with_bands(4);
                ix.add(1u8, [base]);
                assert!(
                    ix.query([near]).contains(&1u8),
                    "{base:08x} vs {near:08x} differ by <= 2 bits and must share a band"
                );
            }
        }
    }

    #[test]
    fn audio_band_counts_are_checked_against_the_u32_width() {
        // 8 bands over 32 bits is 4 bits each: legal, unlike on the 64-bit index where it is 8.
        let _: AudioCorpusIndex<u8> = AudioCorpusIndex::with_bands(8);
    }

    #[test]
    #[should_panic(expected = "bands must divide 32")]
    fn an_audio_band_count_that_does_not_divide_the_width_panics() {
        let _: AudioCorpusIndex<u8> = AudioCorpusIndex::with_bands(3);
    }

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

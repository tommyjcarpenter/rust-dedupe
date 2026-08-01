# perceptual_dedupe

Perceptual near-duplicate detection for video clips and images, in pure Rust.

It finds files that *look* the same even when they are not byte-identical:
re-encodes, resolution changes, trims, and format conversions. The default
build has zero dependencies; decoding and extraction are behind feature flags.

## What it does

- **Video / clip matching.** A clip is sampled into a sequence of 64-bit
  difference hashes (one per frame). Two clips are compared by sliding one
  sequence against the other and taking the alignment with the lowest average
  Hamming distance per overlapping frame. The slide handles trim offsets, so a
  clip that is a trimmed sub-section of another still matches.
- **Motion gating.** Only frames where the content is actually moving are
  scored, so two clips that share nothing but a static watermark or end card do
  not get falsely matched. (Configurable, and switchable off.)
- **Audio corroboration.** An independent 32-bit acoustic sub-fingerprint
  signal (Chromaprint-style) can confirm borderline visual matches — useful
  when a re-encode changes the pixels (color grading, crop, letterboxing) but
  not the audio.
- **Image matching.** A 256-bit (16x16) difference hash with a multi-probe
  pigeonhole index for fast lookup across large sets.
- **Clustering.** Pairwise matches are grouped into duplicate sets with
  union-find.

## Install

```toml
[dependencies]
perceptual_dedupe = "0.1"
```

Optional features (all off by default):

| Feature | Pulls in | Enables |
|---------|----------|---------|
| `serde` | `serde` | `Serialize`/`Deserialize` on the result and config types |
| `file-hash` | `sha2` | `file_hash::hash_file` — a SHA-256 exact-file digest |
| `ffmpeg` | nothing (just `std::process`) | `extract::Extractor` — frame + audio-fingerprint extraction via `ffmpeg`/`fpcalc` |
| `image` | `image` | decode an image file/bytes for the 256-bit hash |

```toml
perceptual_dedupe = { version = "0.1", features = ["ffmpeg", "image"] }
```

## Usage

For a complete, runnable end-to-end demo — hashing, matching trims/re-encodes,
clustering, and the image path — see [`examples/workflow.rs`](examples/workflow.rs):

```sh
cargo run --example workflow
```

### Compare two video clips

```rust
use perceptual_dedupe::{DedupParams, best_alignment};

// Each clip is a sequence of per-frame 64-bit hashes. Produce them yourself,
// or with the `ffmpeg` feature (below).
let clip_a: Vec<u64> = /* frame hashes */ vec![];
let clip_b: Vec<u64> = /* frame hashes */ vec![];

let params = DedupParams::default();
let m = best_alignment(&clip_a, &clip_b, params.min_overlap_frames, params.motion_bits);
if m.matched() && m.avg_bits <= params.threshold_bits {
    println!("duplicate: avg {:.1} bits over {} frames", m.avg_bits, m.overlap);
}
```

`Alignment::classify` reports how the clips overlap (identical, one contains the
other, or partial), which is what you need to decide "keep the longer copy".

### Find duplicates across many clips

```rust
use std::collections::{HashMap, HashSet};
use perceptual_dedupe::{DedupParams, find_candidates, cluster_edges};

// id -> that clip's frame hashes
let clips: HashMap<u64, Vec<u64>> = /* ... */ HashMap::new();
// id -> rank; the lower-ranked clip in a pair becomes the group's "canonical"
let ranks: HashMap<u64, u64> = /* ... */ HashMap::new();

let params = DedupParams::default();
let edges = find_candidates(&clips, &ranks, &HashSet::new(), &params);
let pairs: Vec<(u64, u64)> = edges.iter().map(|e| e.pair()).collect();
for group in cluster_edges(&pairs) {
    println!("duplicate set: {group:?}");
}
```

### Match one clip against a large corpus

`find_candidates` answers "which pairs in this set are dupes" in one all-pairs
pass. When you instead have a big, persistent corpus and want "which corpus
items could match THIS incoming clip", build a [`FrameCorpusIndex`] once and
query it per clip — a pigeonhole band prefilter that returns a small candidate
set to verify, instead of rescanning the whole corpus.

```rust
use perceptual_dedupe::{FrameCorpusIndex, DedupParams, score_visual_segments};

// Build the index once over the corpus (each item's frame hashes).
let mut index = FrameCorpusIndex::new();
index.add("archived_clip_1", [0x1122_3344_5566_7788_u64, 0x99AA_BBCC_DDEE_FF00]);
index.add("archived_clip_2", [0x0000_0000_0000_0001_u64]);

// Per incoming clip: get candidates, then confirm each with a full score.
let incoming: Vec<u64> = vec![0x1122_3344_5566_7788];
let params = DedupParams::default();
for id in index.query(incoming.iter().copied()) {
    // score_visual_segments(&[incoming.clone()], &corpus_segments_for(id), &params)
    //     .filter(|s| s.avg_bits <= params.threshold_bits) => a real match
    let _ = id;
}
```

The index errs toward recall (extra candidates); the score pass is what actually
accepts or rejects. Append-only — rebuild to forget an item.

### Dedup images

```rust
use perceptual_dedupe::{ImageHash, find_duplicates};

// `from_image_path` needs the `image` feature; without it, build hashes from
// your own grayscale pixels via `ImageHash::from_gray_rows`. It converts to
// grayscale with the BT.601 weights, so hashes stay comparable with those from
// other toolchains rather than depending on a decoder's default standard.
let hashes: Vec<ImageHash> = vec![
    ImageHash::from_image_path("a.jpg").unwrap(),
    ImageHash::from_image_path("b.jpg").unwrap(),
];

// Group images within 10 bits (out of 256) of each other; each group is a list
// of indices into `hashes`. Picking which one to keep is up to you.
for group in find_duplicates(&hashes, 10) {
    println!("duplicate set: {group:?}");
}
```

### Extract frame hashes from a file (`ffmpeg` feature)

```rust
use std::path::Path;
use perceptual_dedupe::extract::Extractor;

// Samples the file at a fixed low frame rate, scales each frame to 9x8
// grayscale, and difference-hashes it. Needs `ffmpeg` on PATH.
let frames = Extractor::default().frame_hashes(Path::new("clip.mp4"), 0.0, 60, None)?;
# Ok::<(), perceptual_dedupe::extract::ExtractError>(())
```

## Tuning

Everything that affects matching lives in `DedupParams`: the sample window,
the visual/audio thresholds, the overlap floors, and `motion_bits`. Setting
`motion_bits = 0` turns the motion gate off and scores a plain average over
every overlapping frame. `audio_motion_bits` is the same gate on the audio
signal, off by default; silence fingerprints to a constant run, so without it
two silent stretches align at ~0 bits and match anything.

`audio_alone_bits` is off by default and opts into a third verdict tier, in
which audio admits a pair on its own. A frame hash cannot see through a
reframe — a de-pillarboxed crop and a smaller copy of the same clip read ~20
of 64 bits apart — so for those pairs the visual ceiling bounds noise and
nothing admits them.

Set it far tighter than `audio_threshold_bits`: admitting a pair alone is a
higher bar than corroborating one already in range. Clips can share a
soundtrack over unrelated footage, so a consumer that deletes should treat
`DupeVerdict::AudioNearIdentical` as review-not-delete. Turning the tier on
also widens `needs_audio_corroboration`, so expect more audio alignments.

The one thing that is *not* tunable is the pixel-side sampling geometry
(`SAMPLE_W` = 9, `SAMPLE_H` = 8, `SAMPLE_FPS` = 6). `dhash_9x8` and those
constants define the exact bits of the stored hash, so they are fixed: hashes
computed by any version of the crate stay comparable.

## Candidate indexes

`FrameCorpusIndex` answers "which items in a large corpus could match this one" without
rescanning per query — a pigeonhole band index, so two hashes within `bands - 1` bits are
guaranteed to share a band. It is a prefilter only: verify every candidate with a real score.

It is generic over the hash width. `AudioCorpusIndex` is the same index over `u32` audio
sub-fingerprints, and is usually the one worth building: a frame hash cannot see through a
reframe, so on re-encoded content no two frames are near-identical and a visual index surfaces
nothing, while audio reads near-zero on shared content and does land in the same bands.

`query` unions buckets, which is the right call for a short query. Over a long one it stops
selecting: with a few hundred elements across several bands, a chance collision against any given
candidate is all but certain, so almost the whole corpus comes back. `query_votes` checks the
actual distance and counts how many distinct query elements have a near match, which restores
selectivity — near-identity is rare where collision is not. Within 3 bits of a 32-bit hash lie
5489 of 2^32 values, so an accidental vote runs about 1.3e-6 per element pair while genuinely
shared content votes once per shared element.

Keep `max_bits` under the pigeonhole limit (`bands - 1`) or recall breaks: a pair farther apart
than that need not share a band, so it is never in a bucket to be checked at all.

## Partial overlap

`best_alignment` answers "are these the same clip": it averages over the whole
intersection, so two clips that share a scene and then diverge have the shared
part averaged with the part that isn't, and a real overlap disappears into the
mean. No threshold recovers it — the number measures the wrong span.

`best_matching_run`, and `best_matching_audio_run` for the audio side, answer
"do these share footage, and which part". At each offset they find the best
window holding exactly `min_overlap` scored elements, and return it alongside
the whole-overlap mean at that same offset. A low `run_avg_bits` beside a high
`full_avg_bits` is precisely a partial overlap. Cost is the same
`O(len_a * len_b)` sweep as the plain alignment.

Three cautions. The scan always returns its best window, so the DISTANCE is
what rejects a pair, never the absence of a result — a caller treating `Some`
as a match would flag everything. Keep the motion gate on, because a shared
black card is exactly the shape the scan hunts for. And do not feed run matches
into union-find clustering: connected-component collapse chains unrelated clips
together through a shared intro.

## Development

```sh
cargo fmt --all --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features
```

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at
your option.

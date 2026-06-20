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

### Dedup images

```rust
use perceptual_dedupe::{ImageHash, find_duplicates};

// `from_image_path` needs the `image` feature; without it, build hashes from
// your own grayscale pixels via `ImageHash::from_gray_rows`.
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
every overlapping frame.

The one thing that is *not* tunable is the pixel-side sampling geometry
(`SAMPLE_W` = 9, `SAMPLE_H` = 8, `SAMPLE_FPS` = 6). `dhash_9x8` and those
constants define the exact bits of the stored hash, so they are fixed: hashes
computed by any version of the crate stay comparable.

## Development

```sh
cargo fmt --all --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --all-features
```

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at
your option.

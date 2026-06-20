//! End-to-end toy workflow: hash, match, and cluster near-duplicates.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example workflow
//! ```
//!
//! Everything here is synthetic — no real media, no files. It shows the shape
//! of a real pipeline: turn content into hash sequences, compare them, and
//! group the duplicates. In production the frame hashes come from
//! `dhash_9x8` over decoded frames (or the `ffmpeg` feature's `Extractor`);
//! here we fabricate them so the example is dependency-free and deterministic.

use std::collections::{HashMap, HashSet};

use perceptual_dedupe::image_hash::GRAY_LEN;
use perceptual_dedupe::{
    DedupParams, ImageHash, OverlapKind, best_alignment, cluster_edges, dhash_9x8, find_candidates,
    find_duplicates,
};

/// A "moving" frame-hash sequence of `frames` hashes. Consecutive frames differ
/// by many bits, so they clear the motion gate the way real footage does.
fn synth_clip(seed: u64, frames: usize) -> Vec<u64> {
    (0..frames as u64)
        .map(|i| seed.wrapping_add(i).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect()
}

/// Simulate a re-encode: every frame picks up the same few bits of noise, so
/// the average distance from the original is `bits_per_frame`.
fn reencode(clip: &[u64], bits_per_frame: u32) -> Vec<u64> {
    // Low `bits_per_frame` bits set, total for any value (a >= 64 shift would
    // otherwise panic).
    let mask = if bits_per_frame >= 64 {
        u64::MAX
    } else {
        (1u64 << bits_per_frame) - 1
    };
    clip.iter().map(|&h| h ^ mask).collect()
}

/// A non-uniform 17x16 grayscale image (the bytes `ImageHash` expects).
fn gray_image(seed: u64) -> Vec<u8> {
    (0..GRAY_LEN as u64)
        .map(|i| (seed.wrapping_add(i).wrapping_mul(2_654_435_761) >> 3) as u8)
        .collect()
}

fn main() {
    // ----- Step 0: a single frame -> a single 64-bit hash --------------------
    // A 9x8 grayscale frame is 72 bytes. `dhash_9x8` reduces it to a u64.
    let frame: Vec<u8> = (0..72).map(|i| (i as u8).wrapping_mul(11)).collect();
    println!(
        "one frame (72 bytes) -> dhash_9x8 = {:#018x}\n",
        dhash_9x8(&frame)
    );

    // ----- Step 1: build a few clips as hash sequences -----------------------
    // Each clip is `Vec<u64>` (one hash per sampled frame). We model a small
    // library: an original, an exact re-import, a re-encode, a trimmed
    // sub-section, and an unrelated clip.
    let original = synth_clip(1, 60);
    let clips: HashMap<&str, Vec<u64>> = HashMap::from([
        ("original", original.clone()),
        ("reimport", original.clone()), // byte-different but pixel-identical
        ("reencode", reencode(&original, 3)), // ~3 bits/frame of noise
        ("trimmed", original[20..].to_vec()), // last 40 frames of the original
        ("unrelated", synth_clip(999, 60)), // different content
    ]);

    // Rank decides which clip in a matched pair is the "canonical" (kept) one;
    // lower rank wins. Here we prefer the original, then its variants.
    let ranks: HashMap<&str, u64> = HashMap::from([
        ("original", 0),
        ("reimport", 1),
        ("reencode", 2),
        ("trimmed", 3),
        ("unrelated", 4),
    ]);

    let params = DedupParams::default();

    // ----- Step 2: find candidate duplicate pairs (all-pairs) ----------------
    let mut edges = find_candidates(&clips, &ranks, &HashSet::new(), &params);
    edges.sort_by(|a, b| (a.canonical, a.member).cmp(&(b.canonical, b.member)));

    println!("duplicate pairs (avg bits over the aligned overlap):");
    for e in &edges {
        println!(
            "  {:<9} is a duplicate of {:<9}  avg {:>4.1} bits over {} frames",
            e.member, e.canonical, e.avg_bits, e.overlap_frames
        );
    }

    // ----- Step 3: cluster the pairs into duplicate sets ---------------------
    let pairs: Vec<(&str, &str)> = edges.iter().map(|e| e.pair()).collect();
    let mut groups = cluster_edges(&pairs);
    for g in &mut groups {
        g.sort_unstable();
    }
    groups.sort();

    println!("\nduplicate sets (clips with no match are not listed):");
    for (n, group) in groups.iter().enumerate() {
        println!("  set {}: {:?}", n + 1, group);
    }

    // ----- Step 4: subset vs superset for one pair ---------------------------
    // The alignment also tells you *how* two clips overlap, so a caller can
    // keep the longer copy.
    let al = best_alignment(
        &original,
        &clips["trimmed"],
        params.min_overlap_frames,
        params.motion_bits,
    );
    let kind = al.classify(original.len(), clips["trimmed"].len());
    println!(
        "\n'original' vs 'trimmed': shift {}, overlap {} -> {:?}",
        al.shift, al.overlap, kind
    );
    assert_eq!(
        kind,
        OverlapKind::Contains,
        "trimmed is wholly inside original"
    );

    // ----- Step 5: the image path --------------------------------------------
    // Still images use a 256-bit hash. We make one image, a near-duplicate of
    // it (a handful of flipped bits), and an unrelated image.
    let img = ImageHash::from_gray_rows(&gray_image(7)).unwrap();
    let img_near = {
        let mut bytes = *img.as_bytes();
        bytes[0] ^= 0b0000_0111; // flip 3 bits
        bytes[20] ^= 0b0000_0011; // flip 2 more -> distance 5
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        ImageHash::from_hex(&hex).unwrap()
    };
    let img_other = ImageHash::from_gray_rows(&gray_image(200)).unwrap();

    let images = ["photo", "photo_resaved", "different_photo"];
    let hashes = [img, img_near, img_other];
    println!(
        "\nimage distances from 'photo': resaved={} bits, different={} bits",
        img.hamming(&img_near),
        img.hamming(&img_other)
    );

    println!("image duplicate sets within 10 bits:");
    for group in find_duplicates(&hashes, 10) {
        let names: Vec<&str> = group.iter().map(|&i| images[i]).collect();
        println!("  {names:?}");
    }
}

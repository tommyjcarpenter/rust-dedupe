//! 256-bit image difference hash, pigeonhole index, and clustering parity.
//!
//! Vectors mirror the reference image-dedup tests. Hashes are built from hex
//! (or grayscale rows) so they are exact and need no image decoder.

use perceptual_dedupe::image_hash::{
    GRAY_LEN, HASH_BYTES, HammingIndex, ImageHash, ROW_STRIDE, find_duplicates,
};

fn hash_with_low_bits(bits: u32) -> ImageHash {
    // A 256-bit value with the lowest `bits` bits set (bits <= 64).
    let value: u64 = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let mut hex = "0".repeat(48);
    hex.push_str(&format!("{value:016x}"));
    ImageHash::from_hex(&hex).unwrap()
}

#[test]
fn all_zero_and_all_one_hashes() {
    let zero = ImageHash::from_hex(&"0".repeat(HASH_BYTES * 2)).unwrap();
    let ones = ImageHash::from_hex(&"f".repeat(HASH_BYTES * 2)).unwrap();
    assert_eq!(zero.hamming(&zero), 0);
    assert_eq!(zero.hamming(&ones), 256);
    assert_eq!(ones.hamming(&ones), 0);
}

#[test]
fn hamming_counts_low_bits() {
    let zero = hash_with_low_bits(0);
    assert_eq!(zero.hamming(&hash_with_low_bits(13)), 13);
    assert_eq!(zero.hamming(&hash_with_low_bits(14)), 14);
}

#[test]
fn two_identical_hashes_form_one_group() {
    let h = hash_with_low_bits(20);
    let groups = find_duplicates(&[h, h], 13);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].len(), 2);
}

#[test]
fn distant_hashes_do_not_group() {
    let zero = hash_with_low_bits(0);
    let ones = ImageHash::from_hex(&"f".repeat(HASH_BYTES * 2)).unwrap();
    assert!(find_duplicates(&[zero, ones], 13).is_empty());
}

#[test]
fn two_separate_groups() {
    let a0 = hash_with_low_bits(0);
    let a1 = hash_with_low_bits(2); // distance 2 from a0
    let b0 = ImageHash::from_hex(&"f".repeat(HASH_BYTES * 2)).unwrap();
    // b1 differs from b0 by 2 bits (clear the lowest two nibbles' low bits).
    let mut b1_hex = "f".repeat(HASH_BYTES * 2);
    b1_hex.replace_range(HASH_BYTES * 2 - 1.., "c"); // last nibble f->c flips 2 bits
    let b1 = ImageHash::from_hex(&b1_hex).unwrap();

    let groups = find_duplicates(&[a0, a1, b0, b1], 4);
    assert_eq!(groups.len(), 2);
    for g in &groups {
        assert_eq!(g.len(), 2);
    }
}

#[test]
fn threshold_boundary_groups_at_and_below() {
    let base = hash_with_low_bits(0);
    let at = hash_with_low_bits(13);
    let beyond = hash_with_low_bits(14);
    assert_eq!(find_duplicates(&[base, at], 13).len(), 1);
    assert!(find_duplicates(&[base, beyond], 13).is_empty());
}

#[test]
fn index_has_no_false_negative_within_threshold() {
    // A single-bit flip must surface as a candidate (pigeonhole guarantee).
    let base = hash_with_low_bits(40);
    let mut bytes = *base.as_bytes();
    bytes[7] ^= 0b0001_0000; // flip one bit somewhere
    let flipped = ImageHash::from_hex(&{
        let mut s = String::new();
        for b in bytes {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
        }
        s
    })
    .unwrap();
    assert_eq!(base.hamming(&flipped), 1);

    let index = HammingIndex::build(&[base]);
    assert!(index.query(&flipped, None).contains(&0));
}

#[test]
fn index_finds_exact_matches() {
    let hashes: Vec<ImageHash> = (0..20).map(hash_with_low_bits).collect();
    let index = HammingIndex::build(&hashes);
    for (i, h) in hashes.iter().enumerate() {
        assert!(index.query(h, None).contains(&i));
    }
}

#[test]
fn from_gray_rows_round_trips_through_hex() {
    let gray: Vec<u8> = (0..GRAY_LEN).map(|i| ((i * 7) % 256) as u8).collect();
    assert_eq!(gray.len(), ROW_STRIDE * 16);
    let h = ImageHash::from_gray_rows(&gray).unwrap();
    assert_eq!(ImageHash::from_hex(&h.to_hex()).unwrap(), h);
}

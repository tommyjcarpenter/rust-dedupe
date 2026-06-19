//! Bit-parity vectors for the 9x8 difference hash.
//!
//! These pin the exact `(72 grayscale bytes -> u64)` mapping. The hash is
//! persisted by consumers, so any change here is a corpus-invalidating bug —
//! the vectors must never need editing.

use perceptual_dedupe::frame_hash::{SAMPLE_BYTES, dhash_9x8};

/// Build a 72-byte frame from a per-pixel closure of (row, col).
fn frame(mut f: impl FnMut(usize, usize) -> u8) -> Vec<u8> {
    let mut out = vec![0u8; SAMPLE_BYTES];
    for row in 0..8 {
        for col in 0..9 {
            out[row * 9 + col] = f(row, col);
        }
    }
    out
}

#[test]
fn solid_frames_hash_to_zero() {
    for value in [0u8, 1, 64, 128, 200, 255] {
        assert_eq!(dhash_9x8(&[value; SAMPLE_BYTES]), 0);
    }
}

#[test]
fn strictly_decreasing_rows_are_all_ones() {
    // Every adjacent pair has left > right, so all 64 bits set.
    let gray = frame(|_, col| (200 - col as isize * 10) as u8);
    assert_eq!(dhash_9x8(&gray), 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn strictly_increasing_rows_are_zero() {
    // Every adjacent pair has left < right, so no bits set.
    let gray = frame(|_, col| (10 + col * 10) as u8);
    assert_eq!(dhash_9x8(&gray), 0);
}

#[test]
fn single_bit_at_row0_col0() {
    // Only pixel (0,0) exceeds its right neighbor; bit (0*8 + 0) = bit 0.
    let gray = frame(|row, col| if row == 0 && col == 0 { 2 } else { 1 });
    assert_eq!(dhash_9x8(&gray), 1);
}

#[test]
fn low_byte_set_by_decreasing_first_row() {
    // Row 0 strictly decreasing -> bits 0..=7 (low byte 0xFF); other rows flat.
    let gray = frame(|row, col| if row == 0 { (9 - col) as u8 } else { 5 });
    assert_eq!(dhash_9x8(&gray), 0x0000_0000_0000_00FF);
}

#[test]
fn second_byte_set_by_decreasing_second_row() {
    // Row 1 strictly decreasing -> bits 8..=15 (0xFF00).
    let gray = frame(|row, col| if row == 1 { (9 - col) as u8 } else { 5 });
    assert_eq!(dhash_9x8(&gray), 0x0000_0000_0000_FF00);
}

#[test]
fn top_byte_set_by_decreasing_last_row() {
    // Row 7 strictly decreasing -> bits 56..=63 (0xFF00_0000_0000_0000).
    let gray = frame(|row, col| if row == 7 { (9 - col) as u8 } else { 5 });
    assert_eq!(dhash_9x8(&gray), 0xFF00_0000_0000_0000);
}

#[test]
fn deterministic_for_repeated_input() {
    let gray = frame(|row, col| ((row * 9 + col) as u8).wrapping_mul(13).wrapping_add(7));
    assert_eq!(dhash_9x8(&gray), dhash_9x8(&gray));
}

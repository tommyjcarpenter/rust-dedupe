//! 64-bit difference hash for a 9x8 grayscale frame, plus the fixed sampling
//! geometry that defines its input.
//!
//! # Bit-parity invariant
//!
//! [`dhash_9x8`] is the load-bearing primitive of the whole crate. Its output
//! is persisted by consumers (potentially across millions of rows), so its bit
//! layout MUST never change: a single flipped bit would make every stored hash
//! incomparable with newly computed ones and silently break matching.
//!
//! The sampling constants below define the exact bytes fed to the hash. They
//! are fixed, never tunable — see [`crate::config`] for the rationale.

/// Sampled frame width in pixels. A 9th column is deliberate: it yields the
/// 8th column of differences, bringing the hash to exactly 64 bits.
pub const SAMPLE_W: u32 = 9;

/// Sampled frame height in pixels.
pub const SAMPLE_H: u32 = 8;

/// Bytes per sampled frame: one grayscale byte per pixel, row-major.
pub const SAMPLE_BYTES: usize = (SAMPLE_W as usize) * (SAMPLE_H as usize);

/// Frames sampled per second of source.
pub const SAMPLE_FPS: u32 = 6;

/// 9x8 difference hash. Bit `(row * 8 + col)` is set iff
/// `gray[row * 9 + col] > gray[row * 9 + col + 1]`.
///
/// `gray` is exactly [`SAMPLE_BYTES`] (72) bytes, row-major with a stride of
/// [`SAMPLE_W`] (9). The 9 columns give 8 adjacent-pixel comparisons per row,
/// and 8 rows give exactly 64 bits.
///
/// # Panics
///
/// Debug builds assert `gray.len() == SAMPLE_BYTES`. In release builds a
/// shorter slice will panic on out-of-bounds indexing; callers must pass a
/// correctly sized frame.
pub fn dhash_9x8(gray: &[u8]) -> u64 {
    debug_assert_eq!(gray.len(), SAMPLE_BYTES);
    let mut h: u64 = 0;
    for row in 0..8 {
        let base = row * 9;
        for col in 0..8 {
            if gray[base + col] > gray[base + col + 1] {
                h |= 1u64 << (row * 8 + col);
            }
        }
    }
    h
}

/// Hamming distance between two 64-bit frame hashes: the number of differing
/// bits.
#[inline]
pub fn hamming64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_frame_hashes_to_zero() {
        for value in [0u8, 128, 255] {
            assert_eq!(dhash_9x8(&[value; SAMPLE_BYTES]), 0);
        }
    }

    #[test]
    fn identical_inputs_identical_hashes() {
        let frame: Vec<u8> = (0..SAMPLE_BYTES)
            .map(|i| (i as u8).wrapping_mul(13).wrapping_add(7))
            .collect();
        assert_eq!(dhash_9x8(&frame), dhash_9x8(&frame));
    }

    #[test]
    fn different_inputs_differ() {
        let a: Vec<u8> = (0..SAMPLE_BYTES)
            .map(|i| (i as u8).wrapping_mul(13))
            .collect();
        let b: Vec<u8> = (0..SAMPLE_BYTES)
            .map(|i| (i as u8).wrapping_mul(13).wrapping_add(1))
            .collect();
        assert_ne!(dhash_9x8(&a), dhash_9x8(&b));
    }

    #[test]
    fn hamming64_basics() {
        assert_eq!(hamming64(0, 0), 0);
        assert_eq!(hamming64(0, u64::MAX), 64);
        for bit in [0u32, 7, 31, 63] {
            assert_eq!(hamming64(0, 1 << bit), 1);
        }
    }
}

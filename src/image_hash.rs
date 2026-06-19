//! 256-bit (16x16) difference hash for still images, with a multi-probe
//! pigeonhole index for fast Hamming lookup and union-find clustering.
//!
//! The video path uses a tiny 9x8 hash because it has many frames per clip to
//! average over; a single still image has only itself, so it uses a much
//! larger 16x16 hash (256 bits) for precision.
//!
//! # Parity scope
//!
//! [`ImageHash::from_gray_rows`] is bit-exact: given the same grayscale pixels
//! it produces the same bits as the reference implementation, byte for byte,
//! and [`ImageHash::to_hex`] matches a zero-padded 64-char hex of that value.
//! Decoding and resizing an actual image file (the `image` feature) is NOT
//! bit-exact across libraries, because resampling filters differ; parity is
//! guaranteed only from the grayscale rows inward.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::cluster::cluster_edges;

/// Side length of the hash grid. The hash is `HASH_SIZE * HASH_SIZE` bits.
pub const HASH_SIZE: usize = 16;

/// Pixels per row fed to the hash: one extra column yields the last column of
/// differences.
pub const ROW_STRIDE: usize = HASH_SIZE + 1;

/// Grayscale pixels required by [`ImageHash::from_gray_rows`]:
/// `ROW_STRIDE * HASH_SIZE`, row-major.
pub const GRAY_LEN: usize = ROW_STRIDE * HASH_SIZE;

/// Bytes in the packed hash: `HASH_SIZE * HASH_SIZE / 8`.
pub const HASH_BYTES: usize = HASH_SIZE * HASH_SIZE / 8;

/// Number of chunks the multi-probe index splits a hash into. With 16 chunks,
/// any two hashes within Hamming distance 15 share at least one identical
/// chunk (pigeonhole), so the index has no false negatives at that threshold.
pub const NUM_CHUNKS: usize = HASH_SIZE;

/// A 256-bit image difference hash, stored big-endian (byte 0 is the most
/// significant), so [`ImageHash::to_hex`] reads like a plain hex number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageHash([u8; HASH_BYTES]);

/// Errors from constructing an [`ImageHash`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageHashError {
    /// The grayscale slice was not [`GRAY_LEN`] bytes, or the hex was not
    /// `HASH_BYTES * 2` chars.
    WrongLength {
        /// Expected length.
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// A hex string contained a non-hex character.
    BadHex,
    /// The image crate failed to decode or process the input.
    #[cfg(feature = "image")]
    Decode(String),
}

impl fmt::Display for ImageHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageHashError::WrongLength { expected, got } => {
                write!(f, "wrong length: expected {expected}, got {got}")
            }
            ImageHashError::BadHex => write!(f, "invalid hex digit"),
            #[cfg(feature = "image")]
            ImageHashError::Decode(msg) => write!(f, "image decode failed: {msg}"),
        }
    }
}

impl std::error::Error for ImageHashError {}

impl ImageHash {
    /// Compute the difference hash from [`GRAY_LEN`] grayscale bytes, row-major
    /// with a stride of [`ROW_STRIDE`]. Bit `(y * HASH_SIZE + x)`, counted from
    /// the most significant bit, is set iff `gray[y*ROW_STRIDE + x] >
    /// gray[y*ROW_STRIDE + x + 1]`.
    pub fn from_gray_rows(gray: &[u8]) -> Result<ImageHash, ImageHashError> {
        if gray.len() != GRAY_LEN {
            return Err(ImageHashError::WrongLength {
                expected: GRAY_LEN,
                got: gray.len(),
            });
        }
        let mut bytes = [0u8; HASH_BYTES];
        let mut p = 0usize;
        for y in 0..HASH_SIZE {
            let base = y * ROW_STRIDE;
            for x in 0..HASH_SIZE {
                if gray[base + x] > gray[base + x + 1] {
                    bytes[p / 8] |= 1u8 << (7 - (p % 8));
                }
                p += 1;
            }
        }
        Ok(ImageHash(bytes))
    }

    /// The raw big-endian bytes.
    pub fn as_bytes(&self) -> &[u8; HASH_BYTES] {
        &self.0
    }

    /// Lowercase, zero-padded hex (`HASH_BYTES * 2` chars).
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(HASH_BYTES * 2);
        for byte in self.0 {
            use std::fmt::Write as _;
            let _ = write!(s, "{byte:02x}");
        }
        s
    }

    /// Parse a `HASH_BYTES * 2`-char hex string.
    pub fn from_hex(hex: &str) -> Result<ImageHash, ImageHashError> {
        if hex.len() != HASH_BYTES * 2 {
            return Err(ImageHashError::WrongLength {
                expected: HASH_BYTES * 2,
                got: hex.len(),
            });
        }
        let mut bytes = [0u8; HASH_BYTES];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|_| ImageHashError::BadHex)?;
        }
        Ok(ImageHash(bytes))
    }

    /// Number of differing bits between two hashes.
    pub fn hamming(&self, other: &ImageHash) -> u32 {
        self.0
            .iter()
            .zip(&other.0)
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum()
    }

    /* The 16-bit value of chunk `i` (a 2-byte slice), used by the index. */
    fn chunk(&self, i: usize) -> u16 {
        u16::from_be_bytes([self.0[i * 2], self.0[i * 2 + 1]])
    }

    /// Decode image bytes (JPEG/PNG), convert to grayscale, resize to the hash
    /// grid, and hash. NOT bit-identical to other resamplers — see the module
    /// docs.
    #[cfg(feature = "image")]
    pub fn from_image_bytes(data: &[u8]) -> Result<ImageHash, ImageHashError> {
        let img =
            image::load_from_memory(data).map_err(|e| ImageHashError::Decode(e.to_string()))?;
        let luma = img.to_luma8();
        let resized = image::imageops::resize(
            &luma,
            ROW_STRIDE as u32,
            HASH_SIZE as u32,
            image::imageops::FilterType::Lanczos3,
        );
        ImageHash::from_gray_rows(&resized.into_raw())
    }

    /// Decode an image file and hash it. See [`ImageHash::from_image_bytes`].
    #[cfg(feature = "image")]
    pub fn from_image_path(path: impl AsRef<std::path::Path>) -> Result<ImageHash, ImageHashError> {
        let data = std::fs::read(path).map_err(|e| ImageHashError::Decode(e.to_string()))?;
        ImageHash::from_image_bytes(&data)
    }
}

/// A multi-probe pigeonhole index over many [`ImageHash`]es.
///
/// Each hash is split into [`NUM_CHUNKS`] 16-bit chunks; the index maps each
/// chunk value to the entry indices that share it. A query unions the entries
/// sharing any chunk with the query hash. Callers must still verify each
/// returned candidate with [`ImageHash::hamming`] — the index is a superset
/// filter, not a verdict.
#[derive(Debug, Clone, Default)]
pub struct HammingIndex {
    tables: Vec<HashMap<u16, Vec<usize>>>,
}

impl HammingIndex {
    /// Build an index over `hashes`, keyed by their position in the slice.
    pub fn build(hashes: &[ImageHash]) -> Self {
        let mut index = HammingIndex {
            tables: vec![HashMap::new(); NUM_CHUNKS],
        };
        for (idx, h) in hashes.iter().enumerate() {
            index.add(idx, h);
        }
        index
    }

    /// Add one entry under index `idx`.
    pub fn add(&mut self, idx: usize, hash: &ImageHash) {
        for (i, table) in self.tables.iter_mut().enumerate() {
            table.entry(hash.chunk(i)).or_default().push(idx);
        }
    }

    /// Candidate indices that may be within threshold of `query`. Unions all
    /// entries sharing any chunk; `exclude` drops one index (e.g. the query's
    /// own).
    pub fn query(&self, query: &ImageHash, exclude: Option<usize>) -> HashSet<usize> {
        let mut candidates: HashSet<usize> = HashSet::new();
        for (i, table) in self.tables.iter().enumerate() {
            if let Some(bucket) = table.get(&query.chunk(i)) {
                candidates.extend(bucket.iter().copied());
            }
        }
        if let Some(idx) = exclude {
            candidates.remove(&idx);
        }
        candidates
    }
}

/// Cluster `hashes` into duplicate groups: every pair within `threshold`
/// Hamming bits is an edge, and the connected components are the groups.
///
/// Returns groups of indices into `hashes`; every returned group has at least
/// two members (an item with no near-duplicate is not returned). Choosing
/// which member of a group to keep is the caller's policy, applied to the
/// returned indices.
///
/// The pigeonhole index is zero-false-negative only while `threshold` stays
/// below [`NUM_CHUNKS`] (two hashes within that many bits must share a chunk).
/// For a larger `threshold` the differing bits can spread across every chunk,
/// so this falls back to an exact all-pairs scan to avoid missing pairs.
pub fn find_duplicates(hashes: &[ImageHash], threshold: u32) -> Vec<Vec<usize>> {
    let mut edges: Vec<(usize, usize)> = Vec::new();
    if (threshold as usize) < NUM_CHUNKS {
        let index = HammingIndex::build(hashes);
        for (i, h) in hashes.iter().enumerate() {
            // Sort the candidate indices so the edge order — and thus the
            // grouping order from `cluster_edges` — is deterministic rather
            // than dependent on `HashSet` iteration order.
            let mut candidates: Vec<usize> = index.query(h, Some(i)).into_iter().collect();
            candidates.sort_unstable();
            for j in candidates {
                if j > i && h.hamming(&hashes[j]) <= threshold {
                    edges.push((i, j));
                }
            }
        }
    } else {
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                if hashes[i].hamming(&hashes[j]) <= threshold {
                    edges.push((i, j));
                }
            }
        }
    }
    cluster_edges(&edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_gray_hashes_to_zero() {
        let h = ImageHash::from_gray_rows(&[128u8; GRAY_LEN]).unwrap();
        assert_eq!(h.to_hex(), "0".repeat(HASH_BYTES * 2));
        assert_eq!(h.hamming(&h), 0);
    }

    #[test]
    fn monotonic_decreasing_rows_set_all_bits() {
        // Strictly decreasing across each row => every comparison true => all
        // 256 bits set.
        let mut gray = vec![0u8; GRAY_LEN];
        for y in 0..HASH_SIZE {
            for x in 0..ROW_STRIDE {
                gray[y * ROW_STRIDE + x] = (250 - x * 10) as u8;
            }
        }
        let h = ImageHash::from_gray_rows(&gray).unwrap();
        assert_eq!(h.to_hex(), "f".repeat(HASH_BYTES * 2));
    }

    #[test]
    fn hex_round_trips() {
        let mut gray = vec![0u8; GRAY_LEN];
        for (i, b) in gray.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7);
        }
        let h = ImageHash::from_gray_rows(&gray).unwrap();
        assert_eq!(ImageHash::from_hex(&h.to_hex()).unwrap(), h);
    }

    #[test]
    fn wrong_length_errors() {
        assert!(matches!(
            ImageHash::from_gray_rows(&[0u8; GRAY_LEN - 1]),
            Err(ImageHashError::WrongLength { .. })
        ));
    }

    #[test]
    fn threshold_at_or_above_chunks_uses_exact_scan() {
        // Two hashes that differ by one bit in every chunk: distance NUM_CHUNKS,
        // yet no single chunk is identical, so the pigeonhole index alone would
        // miss them. The exact-scan fallback must still group them.
        let zero = ImageHash::from_hex(&"0".repeat(HASH_BYTES * 2)).unwrap();
        let spread = ImageHash::from_hex(&"8000".repeat(NUM_CHUNKS)).unwrap();
        assert_eq!(zero.hamming(&spread), NUM_CHUNKS as u32);
        // NUM_CHUNKS bits is beyond threshold 15, so not a duplicate there.
        assert!(find_duplicates(&[zero, spread], (NUM_CHUNKS - 1) as u32).is_empty());
        // At/above the chunk count, the fallback finds the pair the index can't.
        let groups = find_duplicates(&[zero, spread], NUM_CHUNKS as u32);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[cfg(feature = "image")]
    #[test]
    fn decodes_solid_png_to_zero_hash() {
        // Encode a synthetic solid-gray image to PNG in memory, then decode and
        // hash it: a uniform image has no pixel differences, so the hash is 0.
        let img = image::GrayImage::from_pixel(32, 32, image::Luma([128u8]));
        let mut bytes: Vec<u8> = Vec::new();
        image::DynamicImage::ImageLuma8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        let h = ImageHash::from_image_bytes(&bytes).unwrap();
        assert_eq!(h.to_hex(), "0".repeat(HASH_BYTES * 2));
    }
}

//! Exact-file content digest (SHA-256).
//!
//! Perceptual hashes catch re-encodes and trims; an exact digest catches the
//! trivial case of the identical bytes imported twice, cheaply and with no
//! false positives. Shared here so consumers do not each reimplement it.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

const CHUNK: usize = 64 * 1024;

/// Stream a file through SHA-256 and return the lowercase hex digest.
pub fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    Ok(hex)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn empty_file_matches_known_digest() {
        let f = NamedTempFile::new().unwrap();
        assert_eq!(
            hash_file(f.path()).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn abc_matches_nist_vector() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"abc").unwrap();
        f.flush().unwrap();
        assert_eq!(
            hash_file(f.path()).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn larger_than_chunk_is_stable() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&vec![0u8; 200 * 1024]).unwrap();
        f.flush().unwrap();
        let h = hash_file(f.path()).unwrap();
        assert_eq!(h.len(), 64);
        assert_eq!(h, hash_file(f.path()).unwrap());
    }
}

//! Frame-hash and audio-fingerprint extraction by shelling out to `ffmpeg`
//! (and `fpcalc` for audio).
//!
//! This is the only part of the crate that touches the outside world, so it is
//! behind the `ffmpeg` feature. It adds nothing but `std::process`: the heavy
//! lifting is done by the external binaries, which the caller must have on
//! PATH (or supply a path to).
//!
//! Both extractors seek with `-ss` before `-i` (a fast keyframe seek). The
//! sub-second imprecision that introduces is harmless: it affects every clip
//! sourced from the same file equally, so pairwise alignment is unchanged.

use std::fmt;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::audio::{AUDIO_CHANNELS, AUDIO_SAMPLE_RATE};
use crate::frame_hash::{SAMPLE_BYTES, SAMPLE_FPS, SAMPLE_H, SAMPLE_W, dhash_9x8};

const READ_BUF: usize = 64 * 1024;

/// Errors from running an extraction.
#[derive(Debug)]
pub enum ExtractError {
    /// Failed to spawn a child process (binary not found, etc.).
    Spawn(std::io::Error),
    /// An IO error reading a child's output.
    Io(std::io::Error),
    /// The caller's cancel flag was set mid-extraction.
    Cancelled,
    /// A child process exited non-zero.
    NonZeroExit(String),
    /// `ffmpeg`'s raw video output was not a whole number of frames.
    Truncated,
    /// A fingerprint value could not be parsed.
    Parse(String),
    /// `fpcalc` produced no `FINGERPRINT=` line (e.g. a silent clip).
    NoFingerprint,
}

impl fmt::Display for ExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtractError::Spawn(e) => write!(f, "spawn process: {e}"),
            ExtractError::Io(e) => write!(f, "read output: {e}"),
            ExtractError::Cancelled => write!(f, "cancelled"),
            ExtractError::NonZeroExit(s) => write!(f, "{s}"),
            ExtractError::Truncated => write!(f, "ffmpeg output truncated mid-frame"),
            ExtractError::Parse(s) => write!(f, "parse fingerprint '{s}'"),
            ExtractError::NoFingerprint => write!(f, "fpcalc produced no fingerprint"),
        }
    }
}

impl std::error::Error for ExtractError {}

#[inline]
fn is_cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|c| c.load(Ordering::Relaxed))
}

/* Drain a child's piped stderr on a background thread, returning a handle whose
join yields the trimmed text. Draining concurrently (rather than after the child
exits) means stderr can never fill its pipe buffer and block the child mid-run —
the classic two-pipe deadlock when the parent reads only stdout. The caller
joins after the child has exited. */
fn spawn_stderr_reader(
    stderr: Option<std::process::ChildStderr>,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buf = String::new();
        if let Some(mut s) = stderr {
            let _ = s.read_to_string(&mut buf);
        }
        buf.trim().to_string()
    })
}

/* Build a `NonZeroExit` that includes the tool's stderr when there is any, so a
production failure (missing codec, bad input, permissions) is diagnosable. */
fn exit_error(tool: &str, status: std::process::ExitStatus, stderr: String) -> ExtractError {
    if stderr.is_empty() {
        ExtractError::NonZeroExit(format!("{tool} exit {status}"))
    } else {
        ExtractError::NonZeroExit(format!("{tool} exit {status}: {stderr}"))
    }
}

/// Runs `ffmpeg` / `fpcalc`, capturing the binary names once so per-call
/// signatures stay short. Construct with [`Extractor::default`] (uses
/// `ffmpeg` and `fpcalc` from PATH) or set the paths explicitly.
#[derive(Debug, Clone)]
pub struct Extractor {
    /// Path or name of the ffmpeg binary.
    pub ffmpeg_bin: String,
    /// Path or name of the fpcalc (Chromaprint) binary.
    pub fpcalc_bin: String,
}

impl Default for Extractor {
    fn default() -> Self {
        Self {
            ffmpeg_bin: "ffmpeg".to_string(),
            fpcalc_bin: "fpcalc".to_string(),
        }
    }
}

impl Extractor {
    /// Sample `path[start_secs .. start_secs + window_secs]` into a sequence of
    /// 64-bit frame hashes: one `ffmpeg` invocation at [`SAMPLE_FPS`], scaled
    /// to [`SAMPLE_W`]x[`SAMPLE_H`] grayscale, hashed per frame.
    ///
    /// `cancel`, if set during the read, kills the child and returns
    /// [`ExtractError::Cancelled`].
    pub fn frame_hashes(
        &self,
        path: &Path,
        start_secs: f32,
        window_secs: u32,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<u64>, ExtractError> {
        let vf = format!("fps={SAMPLE_FPS},scale={SAMPLE_W}:{SAMPLE_H},format=gray");
        let mut child = Command::new(&self.ffmpeg_bin)
            .args(["-nostdin", "-v", "error", "-nostats"])
            .arg("-ss")
            .arg(format!("{start_secs:.3}"))
            .arg("-t")
            .arg(window_secs.to_string())
            .arg("-i")
            .arg(path)
            .args(["-vf", &vf])
            .args(["-an", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(ExtractError::Spawn)?;

        let mut stdout = child.stdout.take().expect("stdout was piped");
        let stderr_reader = spawn_stderr_reader(child.stderr.take());
        // The output size is predictable from the fixed sampling geometry
        // (fps * window seconds frames, each SAMPLE_BYTES), so pre-allocate to
        // avoid repeated reallocation while streaming.
        let expected = (SAMPLE_FPS as usize)
            .saturating_mul(window_secs as usize)
            .saturating_mul(SAMPLE_BYTES);
        let mut data: Vec<u8> = Vec::with_capacity(expected);
        let mut buf = [0u8; READ_BUF];
        loop {
            if is_cancelled(cancel) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ExtractError::Cancelled);
            }
            match stdout.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => data.extend_from_slice(&buf[..n]),
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(ExtractError::Io(e));
                }
            }
        }
        let status = child.wait().map_err(ExtractError::Io)?;
        let stderr = stderr_reader.join().unwrap_or_default();
        if !status.success() {
            return Err(exit_error("ffmpeg", status, stderr));
        }
        if !data.len().is_multiple_of(SAMPLE_BYTES) {
            return Err(ExtractError::Truncated);
        }
        Ok(data.chunks_exact(SAMPLE_BYTES).map(dhash_9x8).collect())
    }

    /// Sample `path[start_secs .. start_secs + window_secs]` into Chromaprint
    /// sub-fingerprints: `ffmpeg` decodes mono [`AUDIO_SAMPLE_RATE`] WAV piped
    /// into `fpcalc -raw`, whose `FINGERPRINT=` line is parsed into `u32`s.
    pub fn audio_fingerprint(
        &self,
        path: &Path,
        start_secs: f32,
        window_secs: u32,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<u32>, ExtractError> {
        let mut ffmpeg = Command::new(&self.ffmpeg_bin)
            .args(["-nostdin", "-v", "error", "-nostats"])
            .arg("-ss")
            .arg(format!("{start_secs:.3}"))
            .arg("-t")
            .arg(window_secs.to_string())
            .arg("-i")
            .arg(path)
            .args([
                "-vn",
                "-ac",
                &AUDIO_CHANNELS.to_string(),
                "-ar",
                &AUDIO_SAMPLE_RATE.to_string(),
            ])
            .args(["-f", "wav", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(ExtractError::Spawn)?;

        let ffmpeg_out = ffmpeg.stdout.take().expect("stdout was piped");
        // Drain ffmpeg's stderr concurrently: we never read it on this thread
        // (we read fpcalc's stdout instead), so without a reader a flood of
        // ffmpeg errors could fill the pipe and block ffmpeg mid-decode.
        let ffmpeg_stderr_reader = spawn_stderr_reader(ffmpeg.stderr.take());
        let mut fpcalc = Command::new(&self.fpcalc_bin)
            .args(["-raw", "-length", "0", "-"])
            .stdin(Stdio::from(ffmpeg_out))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                let _ = ffmpeg.kill();
                let _ = ffmpeg.wait();
                ExtractError::Spawn(e)
            })?;

        let fpcalc_out = fpcalc.stdout.take().expect("stdout was piped");
        let fpcalc_stderr_reader = spawn_stderr_reader(fpcalc.stderr.take());
        let mut fingerprint_line: Option<String> = None;
        for line in BufReader::new(fpcalc_out).lines() {
            if is_cancelled(cancel) {
                let _ = fpcalc.kill();
                let _ = ffmpeg.kill();
                let _ = fpcalc.wait();
                let _ = ffmpeg.wait();
                return Err(ExtractError::Cancelled);
            }
            let line = line.map_err(ExtractError::Io)?;
            if let Some(rest) = line.strip_prefix("FINGERPRINT=") {
                fingerprint_line = Some(rest.to_string());
            }
        }
        let fpcalc_status = fpcalc.wait().map_err(ExtractError::Io)?;
        let ffmpeg_status = ffmpeg.wait().map_err(ExtractError::Io)?;
        let ffmpeg_stderr = ffmpeg_stderr_reader.join().unwrap_or_default();
        let fpcalc_stderr = fpcalc_stderr_reader.join().unwrap_or_default();
        if !ffmpeg_status.success() {
            return Err(exit_error("ffmpeg", ffmpeg_status, ffmpeg_stderr));
        }
        if !fpcalc_status.success() {
            return Err(exit_error("fpcalc", fpcalc_status, fpcalc_stderr));
        }

        let Some(line) = fingerprint_line else {
            return Err(ExtractError::NoFingerprint);
        };
        let mut out = Vec::new();
        for piece in line.split(',') {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            // fpcalc emits signed 32-bit integers; truncate to the low 32 bits
            // so the value round-trips as a u32 sub-fingerprint.
            let parsed: i64 = piece
                .parse()
                .map_err(|_| ExtractError::Parse(piece.to_string()))?;
            out.push(parsed as u32);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn frame_hashes_from_synthetic_clip() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let clip = dir.path().join("synthetic.mp4");
        // A 5-second black 64x64 clip at 6 fps — fully synthetic, no real media.
        let made = Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:rate=6:duration=5",
            ])
            .args([
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-loglevel",
                "error",
            ])
            .arg(&clip)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !made {
            eprintln!("skipping: could not synthesize clip");
            return;
        }
        let hashes = Extractor::default()
            .frame_hashes(&clip, 0.0, 5, None)
            .unwrap();
        // ~6 fps * 5 s; allow slack for encoder framing.
        assert!(hashes.len() >= 20, "got {} frames", hashes.len());
    }
}

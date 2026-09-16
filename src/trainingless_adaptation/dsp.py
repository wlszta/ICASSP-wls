from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class CorruptionSpec:
    cutoff_hz: float
    order: int
    snr_db: float
    noise_reference: str = "filtered"
    filter_mode: str = "causal"


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio)
    if values.ndim == 2:
        values = values.mean(axis=1 if values.shape[1] <= values.shape[0] else 0)
    if values.ndim != 1:
        raise ValueError(f"Expected mono or channelled audio, got {values.shape}")
    return np.ascontiguousarray(values, dtype=np.float32)


def butterworth_lowpass(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    order: int,
    filter_mode: str = "causal",
) -> np.ndarray:
    if not 0 < cutoff_hz < sample_rate / 2:
        raise ValueError("cutoff_hz must be between zero and Nyquist")
    if order < 1:
        raise ValueError("order must be positive")
    if filter_mode not in {"causal", "zero_phase"}:
        raise ValueError(f"Unknown filter mode: {filter_mode}")
    coefficients_b, coefficients_a = signal.butter(order, cutoff_hz / (sample_rate / 2), btype="low")
    clean = to_mono_float32(audio)
    filtered = (
        signal.filtfilt(coefficients_b, coefficients_a, clean)
        if filter_mode == "zero_phase"
        else signal.lfilter(coefficients_b, coefficients_a, clean)
    )
    return np.asarray(filtered, dtype=np.float32)


def add_white_noise(
    filtered: np.ndarray,
    clean: np.ndarray,
    snr_db: float,
    seed: int,
    reference: str,
) -> tuple[np.ndarray, np.ndarray]:
    if reference not in {"filtered", "clean"}:
        raise ValueError(f"Unknown noise reference: {reference}")
    reference_signal = filtered if reference == "filtered" else clean
    reference_power = float(np.mean(np.square(reference_signal, dtype=np.float64)))
    if reference_power <= 0:
        return filtered.copy(), np.zeros_like(filtered)
    noise_power = reference_power / (10.0 ** (snr_db / 10.0))
    generator = np.random.Generator(np.random.PCG64(seed))
    noise = generator.standard_normal(filtered.shape, dtype=np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    noise = noise * np.float32(np.sqrt(noise_power) / max(noise_rms, np.finfo(np.float32).eps))
    return (filtered + noise).astype(np.float32), noise.astype(np.float32)


def corrupt(audio: np.ndarray, sample_rate: int, spec: CorruptionSpec, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = to_mono_float32(audio)
    filtered = butterworth_lowpass(clean, sample_rate, spec.cutoff_hz, spec.order, spec.filter_mode)
    shifted, noise = add_white_noise(filtered, clean, spec.snr_db, seed, spec.noise_reference)
    return shifted, filtered, noise


def measured_snr_db(reference: np.ndarray, noise: np.ndarray) -> float:
    signal_power = float(np.mean(np.square(reference, dtype=np.float64)))
    noise_power = float(np.mean(np.square(noise, dtype=np.float64)))
    return 10.0 * np.log10(signal_power / noise_power)


def mel_cutoff_bin(mel_centers_hz: np.ndarray, cutoff_hz: float) -> int:
    centers = np.asarray(mel_centers_hz, dtype=np.float64)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("mel_centers_hz must be a non-empty vector")
    return int(np.argmin(np.abs(centers - cutoff_hz)))


def map_cutoff_bin(mel_bin: int, mel_bins: int, feature_bins: int) -> int:
    if not 0 <= mel_bin < mel_bins:
        raise ValueError("mel_bin is outside mel range")
    if feature_bins < 1:
        raise ValueError("feature_bins must be positive")
    return min(feature_bins - 1, int(np.floor(feature_bins * mel_bin / mel_bins)))


def kaldi_mel_centers(
    num_bins: int,
    window_length_padded: int,
    sample_frequency: float,
    low_frequency: float,
    high_frequency: float,
    vtln_low: float = 100.0,
    vtln_high: float = -500.0,
    vtln_warp_factor: float = 1.0,
) -> np.ndarray:
    """Return center frequencies from torchaudio's Kaldi mel-bank formula.

    Eq. (5) requires the centers used by the model frontend. PaSST and BEATs
    use Kaldi-compatible banks, so a generic librosa grid can shift the
    cutoff. The vendor settings use a unit VTLN warp; non-unit warps are
    rejected rather than approximated.
    """
    if num_bins <= 3 or window_length_padded <= 0:
        raise ValueError("num_bins must exceed 3 and window length must be positive")
    if vtln_warp_factor != 1.0:
        raise ValueError("Only the unit VTLN warp used by the vendor frontends is supported")
    nyquist = 0.5 * float(sample_frequency)
    low = float(low_frequency)
    high = float(high_frequency)
    if high <= 0.0:
        high += nyquist
    if not 0.0 <= low < high <= nyquist:
        raise ValueError("Invalid Kaldi mel frequency range")

    def mel_scale(frequency: float) -> float:
        return 1127.0 * np.log1p(frequency / 700.0)

    low_mel = mel_scale(low)
    high_mel = mel_scale(high)
    delta = (high_mel - low_mel) / (num_bins + 1)
    centers_mel = low_mel + (np.arange(num_bins, dtype=np.float64) + 1.0) * delta
    return np.asarray(700.0 * np.expm1(centers_mel / 1127.0), dtype=np.float64)

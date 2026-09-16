import numpy as np
import pytest

from trainingless_adaptation.dsp import CorruptionSpec, corrupt, kaldi_mel_centers, measured_snr_db


def test_corruption_is_deterministic_and_has_requested_snr() -> None:
    sample_rate = 44100
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    spec = CorruptionSpec(1000, 4, -10, "filtered")
    first, filtered, noise = corrupt(audio, sample_rate, spec, seed=123)
    second, _, _ = corrupt(audio, sample_rate, spec, seed=123)
    assert np.array_equal(first, second)
    assert abs(measured_snr_db(filtered, noise) - spec.snr_db) < 0.05
    assert first.shape == audio.shape


def test_higher_order_attenuates_high_frequency_more() -> None:
    sample_rate = 44100
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = np.sin(2 * np.pi * 5000 * time).astype(np.float32)
    _, first_order, _ = corrupt(audio, sample_rate, CorruptionSpec(1000, 1, 100), seed=1)
    _, fourth_order, _ = corrupt(audio, sample_rate, CorruptionSpec(1000, 4, 100), seed=1)
    assert np.mean(fourth_order**2) < np.mean(first_order**2)


def test_kaldi_mel_centers_follow_vendor_frontend_ranges() -> None:
    passt = kaldi_mel_centers(128, 1024, 32000, 0, 15000)
    beats = kaldi_mel_centers(128, 512, 16000, 20, 0)
    assert passt.shape == (128,)
    assert beats.shape == (128,)
    assert np.all(np.diff(passt) > 0)
    assert np.all(np.diff(beats) > 0)
    assert 0 < passt[0] < passt[-1] < 15000
    assert 20 < beats[0] < beats[-1] < 8000


def test_zero_phase_filter_is_explicit_and_preserves_requested_snr() -> None:
    sample_rate = 44100
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    spec = CorruptionSpec(1000, 4, -10, "filtered", "zero_phase")
    _, filtered, noise = corrupt(audio, sample_rate, spec, seed=123)
    assert abs(measured_snr_db(filtered, noise) - spec.snr_db) < 0.05
    with pytest.raises(ValueError, match="Unknown filter mode"):
        corrupt(audio, sample_rate, CorruptionSpec(1000, 4, -10, "filtered", "unknown"), seed=123)

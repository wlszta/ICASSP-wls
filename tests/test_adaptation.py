import torch

from trainingless_adaptation.adaptation import TokenGrid, replace_high_frequency_tokens


def test_frequency_major_replacement_preserves_special_tokens() -> None:
    tokens = torch.arange(2 * 8, dtype=torch.float32).reshape(1, 8, 2)
    silent = torch.full_like(tokens, -1)
    grid = TokenGrid(frequency_bins=2, time_bins=3, token_offset=2)
    output = replace_high_frequency_tokens(tokens, silent, grid, cutoff_bin=1, inclusive=True)
    assert torch.equal(output[:, :2], tokens[:, :2])
    patches = output[:, 2:].reshape(1, 2, 3, 2)
    assert torch.equal(patches[:, 0], tokens[:, 2:5].reshape(1, 3, 2))
    assert torch.all(patches[:, 1] == -1)


def test_time_major_sequence_first_replacement() -> None:
    tokens = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 1, 2)
    silent = torch.full_like(tokens, -2)
    grid = TokenGrid(frequency_bins=2, time_bins=3, sequence_first=True, time_major=True)
    output = replace_high_frequency_tokens(tokens, silent, grid, cutoff_bin=1, inclusive=True)
    patches = output.transpose(0, 1).reshape(1, 3, 2, 2)
    assert torch.all(patches[:, :, 1] == -2)
    assert torch.all(patches[:, :, 0] != -2)


def test_repeated_frequency_layout_masks_every_repeat() -> None:
    tokens = torch.zeros(1, 16, 1)
    silent = torch.ones_like(tokens)
    grid = TokenGrid(frequency_bins=2, time_bins=2, frequency_repeats=4)
    output = replace_high_frequency_tokens(tokens, silent, grid, cutoff_bin=1, inclusive=True)
    patches = output.reshape(1, 4, 2, 2, 1)
    assert torch.all(patches[:, :, 0] == 0)
    assert torch.all(patches[:, :, 1] == 1)


def test_exclusive_mask_preserves_cutoff_bin() -> None:
    tokens = torch.zeros(1, 6, 1)
    silent = torch.ones_like(tokens)
    grid = TokenGrid(frequency_bins=3, time_bins=2)
    output = replace_high_frequency_tokens(tokens, silent, grid, cutoff_bin=1, inclusive=False)
    patches = output.reshape(1, 3, 2, 1)
    assert torch.all(patches[:, :2] == 0)
    assert torch.all(patches[:, 2] == 1)

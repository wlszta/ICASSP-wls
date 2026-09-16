from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import torch


@dataclass(frozen=True)
class TokenGrid:
    frequency_bins: int
    time_bins: int
    token_offset: int = 0
    sequence_first: bool = False
    frequency_repeats: int = 1
    time_major: bool = False

    @property
    def patch_tokens(self) -> int:
        return self.frequency_bins * self.time_bins * self.frequency_repeats


def replace_high_frequency_tokens(
    tokens: torch.Tensor,
    silent_tokens: torch.Tensor,
    grid: TokenGrid,
    cutoff_bin: int,
    inclusive: bool,
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected a three-dimensional token tensor, got {tokens.shape}")
    batch_first = tokens.transpose(0, 1) if grid.sequence_first else tokens
    silent_batch = silent_tokens.transpose(0, 1) if grid.sequence_first else silent_tokens
    if silent_batch.shape[0] == 1 and batch_first.shape[0] != 1:
        silent_batch = silent_batch.expand(batch_first.shape[0], -1, -1)
    if silent_batch.shape != batch_first.shape:
        raise ValueError(f"Silent/token shape mismatch: {silent_batch.shape} != {batch_first.shape}")
    end = grid.token_offset + grid.patch_tokens
    if end > batch_first.shape[1]:
        raise ValueError(f"Token grid exceeds sequence length: {end} > {batch_first.shape[1]}")
    start = cutoff_bin if inclusive else cutoff_bin + 1
    start = max(0, min(start, grid.frequency_bins))
    if grid.time_major:
        patches = batch_first[:, grid.token_offset:end].reshape(
            batch_first.shape[0], grid.time_bins, grid.frequency_repeats, grid.frequency_bins, batch_first.shape[-1]
        )
        silent_patches = silent_batch[:, grid.token_offset:end].reshape_as(patches)
        replaced = patches.clone()
        replaced[:, :, :, start:, :] = silent_patches[:, :, :, start:, :]
    else:
        patches = batch_first[:, grid.token_offset:end].reshape(
            batch_first.shape[0], grid.frequency_repeats, grid.frequency_bins, grid.time_bins, batch_first.shape[-1]
        )
        silent_patches = silent_batch[:, grid.token_offset:end].reshape_as(patches)
        replaced = patches.clone()
        replaced[:, :, start:, :, :] = silent_patches[:, :, start:, :, :]
    output = batch_first.clone()
    output[:, grid.token_offset:end] = replaced.reshape(batch_first.shape[0], grid.patch_tokens, batch_first.shape[-1])
    return output.transpose(0, 1) if grid.sequence_first else output


def split_module_output(output: object) -> tuple[torch.Tensor, tuple[object, ...] | None]:
    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], output[1:]
    raise TypeError(f"Unsupported hooked output type: {type(output)!r}")


def join_module_output(tokens: torch.Tensor, remainder: tuple[object, ...] | None) -> object:
    return tokens if remainder is None else (tokens, *remainder)


def capture_silent_outputs(modules: Sequence[torch.nn.Module], forward: Callable[[], object]) -> list[torch.Tensor]:
    captured: list[torch.Tensor | None] = [None] * len(modules)
    handles = []
    for index, module in enumerate(modules):
        def hook(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object, index: int = index) -> None:
            tokens, _ = split_module_output(output)
            captured[index] = tokens.detach().clone()

        handles.append(module.register_forward_hook(hook))
    try:
        forward()
    finally:
        for handle in handles:
            handle.remove()
    if any(tokens is None for tokens in captured):
        raise RuntimeError("Not every adaptation block produced a silent activation")
    return [tokens for tokens in captured if tokens is not None]


@contextmanager
def replacement_hooks(
    modules: Sequence[torch.nn.Module],
    silent_outputs: Sequence[torch.Tensor],
    ending_block: int,
    replace: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
) -> Iterator[None]:
    if ending_block < 0 or ending_block >= len(modules):
        raise ValueError(f"ending_block {ending_block} outside 0..{len(modules) - 1}")
    handles = []
    for index, module in enumerate(modules[: ending_block + 1]):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
            index: int = index,
        ) -> object:
            tokens, remainder = split_module_output(output)
            replaced = replace(index, tokens, silent_outputs[index].to(tokens.device, tokens.dtype))
            return join_module_output(replaced, remainder)

        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

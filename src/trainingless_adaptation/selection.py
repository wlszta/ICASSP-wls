"""Label-free boundary selection utilities for the ICASSP method.

Selector contract: functions under the selector section receive only
calibration probabilities of shape (K, N, C) and method settings. They
never receive labels, true cutoff frequency, family, or evaluation
statistics. Evaluation helpers are separate and may read labels.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PROB_FLOOR = 1e-12
TIE_ATOL = 1e-12
FOLDS = [1, 2, 3, 4, 5]
SEEDS = list(range(1, 31))
N_CLASSES = 50

# Historical split recipes. Do not substitute "the same seed integer"
# for an actually identical permutation.
SPLIT_SS_GRID = 2026082913          # exp13 / analyze_gridc_formal 20/380
SPLIT_SS_BUDGET = 20260901          # exp21 fixed 80/320
CAL_BOOT_SS = 2026091110            # experiment-23 CAL bootstrap stream
PERM_SS = 2026091111
SETRANDOM_SS = 2026091112
UNPAIRED_SS = 2026091113
REPORT_BOOT_SEED = 2026091117
SETRANDOM_DECISION_SEEDS = 10
B_REPORT = 2000
ALPHA_PRIMARY = 0.10
ALPHAS = (0.05, 0.10, 0.20)
MINIMAX_RATIOS = (1.0, 2.0, 4.0)

_FC_RE = re.compile(r"fc(500|1000|2000|4000)")
_SNR_RE = re.compile(r"snr(-?\d+)")


# ---------------------------------------------------------------------------
# Probability / IM
# ---------------------------------------------------------------------------

def validate_probability_tensor(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require a finite, non-negative (K, N, C) tensor with positive row sums.

    This is an input check, not a project-wide gate. Callers that already
    normalized remain valid: row sums of 1 pass.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 3:
        raise RuntimeError(f"probability tensor must be (K, N, C), got {p.shape}")
    if not np.isfinite(p).all():
        raise RuntimeError(f"non-finite probability in tensor of shape {p.shape}")
    if np.any(p < 0):
        raise RuntimeError("negative probability")
    sums = p.sum(axis=2, keepdims=True)
    if np.any(sums <= 0):
        raise RuntimeError("non-positive probability row")
    return p, sums


def normalize_probabilities(p: np.ndarray) -> np.ndarray:
    """float64 + row-wise class normalization (exp13 convention)."""
    p, sums = validate_probability_tensor(p)
    return p / sums


def entropy_last(p: np.ndarray) -> np.ndarray:
    q = np.maximum(p, PROB_FLOOR)
    return -np.sum(q * np.log(q), axis=-1)


def im_scores(p: np.ndarray) -> np.ndarray:
    """IM per candidate for p of shape (K, N, C).

    u_b = H(mean_i p_i^(b)) - mean_i H(p_i^(b)), natural log.
    H(mean p) is recomputed for every input; IM is not a fixed per-clip
    scalar averaged over clips.
    """
    p = np.asarray(p, dtype=np.float64)
    n = float(p.shape[1])
    h_each = entropy_last(p)
    h_cond = np.add.reduce(h_each, axis=1) / n
    p_bar = np.add.reduce(p, axis=1) / n
    h_marg = entropy_last(p_bar)
    return h_marg - h_cond


def im_scores_weighted(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """IM with per-clip multiplicity weights w (counts summing to N)."""
    p = np.asarray(p, dtype=np.float64)
    n = float(w.sum())
    h_each = entropy_last(p)
    h_cond = np.add.reduce(h_each * w[None, :], axis=1) / n
    p_bar = np.add.reduce(p * w[None, :, None], axis=1) / n
    h_marg = entropy_last(p_bar)
    return h_marg - h_cond


def argmax_largest_b(x: np.ndarray, atol: float = TIE_ATOL) -> int:
    """Numerical tie -> largest boundary (formal experiment-23 rule)."""
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    cand = np.flatnonzero(np.isclose(x, m, atol=atol, rtol=0.0))
    return int(cand[-1])


def argmax_first(x: np.ndarray, atol: float = TIE_ATOL) -> int:
    """Legacy tie rule: smallest boundary (replay only)."""
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    cand = np.flatnonzero(np.isclose(x, m, atol=atol, rtol=0.0))
    return int(cand[0])


# ---------------------------------------------------------------------------
# Evaluation (labels allowed here, never inside selectors)
# ---------------------------------------------------------------------------

def accuracy_curve(p: np.ndarray, labels: np.ndarray, idx: np.ndarray,
                   preds: np.ndarray | None = None) -> np.ndarray:
    """Clip-micro accuracy (%) per candidate on an evaluation index set."""
    y = labels[idx]
    if preds is None:
        pred = np.argmax(p[:, idx, :], axis=-1)
    else:
        pred = preds[:, idx]
    return 100.0 * np.mean(pred == y[None, :], axis=1)


def accuracy_curve_macro(p: np.ndarray, labels: np.ndarray, idx: np.ndarray,
                         n_classes: int = N_CLASSES,
                         preds: np.ndarray | None = None) -> np.ndarray:
    """Unweighted per-class mean accuracy (%) on the evaluation index set.

    Classes absent from the evaluation split contribute 0; the divisor is
    always n_classes. This is not the same as clip-micro after random CAL
    removal.
    """
    y = labels[idx]
    if preds is None:
        pred = np.argmax(p[:, idx, :], axis=-1)
    else:
        pred = preds[:, idx]
    correct = pred == y[None, :]
    out = np.zeros(pred.shape[0], dtype=np.float64)
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            out += correct[:, mask].mean(axis=1)
    return 100.0 * out / float(n_classes)


# ---------------------------------------------------------------------------
# Paired bootstrap and admissible set (label-free)
# ---------------------------------------------------------------------------

def _counts_from_idx(idx: np.ndarray, n: int) -> np.ndarray:
    """idx (B, N) -> counts (B, N) with multiplicity. Uses add.at."""
    b = idx.shape[0]
    counts = np.zeros((b, n), dtype=np.float64)
    np.add.at(counts, (np.arange(b, dtype=np.int64)[:, None], idx), 1.0)
    return counts


def paired_bootstrap_im(p_cal: np.ndarray, B: int, rng: np.random.Generator,
                        paired: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """B bootstrap IM vectors.

    With paired=True every candidate shares one clip-index draw per
    replicate, and H(mean p) is recomputed on that draw. With paired=False
    each candidate is resampled independently (correlation ablation only).

    Returns (u_boot (B, K), u_hat (K,)).
    """
    p_cal = np.asarray(p_cal, dtype=np.float64)
    k, n, _ = p_cal.shape
    u_hat = im_scores(p_cal)
    h_each = entropy_last(p_cal)  # (K, N)
    if paired:
        idx = rng.integers(0, n, size=(B, n))
        counts = _counts_from_idx(idx, n)  # (B, N)
        h_cond = (h_each @ counts.T) / float(n)  # (K, B)
        p_bar = np.einsum("knc,bn->kbc", p_cal, counts, optimize=True) / float(n)
        h_marg = entropy_last(p_bar)
        u_boot = (h_marg - h_cond).T
        return u_boot, u_hat
    u_boot = np.empty((B, k), dtype=np.float64)
    for b in range(k):
        idx = rng.integers(0, n, size=(B, n))
        counts = _counts_from_idx(idx, n)
        pb = p_cal[b]
        h_cond = (h_each[b] @ counts.T) / float(n)
        p_bar = (counts @ pb) / float(n)
        h_marg = entropy_last(p_bar)
        u_boot[:, b] = h_marg - h_cond
    return u_boot, u_hat


def paired_bootstrap_im_direct(p_cal: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Direct gather IM for each row of idx (B, N). Used to check weights."""
    p_cal = np.asarray(p_cal, dtype=np.float64)
    out = np.empty((idx.shape[0], p_cal.shape[0]), dtype=np.float64)
    for r in range(idx.shape[0]):
        out[r] = im_scores(p_cal[:, idx[r], :])
    return out


def admissible_set(u_hat: np.ndarray, u_boot: np.ndarray, alpha: float) -> dict:
    """Simultaneous paired-difference admissible set.

    e_r,b = u_boot_r,b - u_hat_b
    t_r = max_b e_r,b - min_b e_r,b
    c = quantile(t, 1-alpha)   (NumPy linear quantile)
    S = {b : max_j u_hat_j - u_hat_b <= c}

    S may be non-contiguous; gaps are not filled. S always contains the
    sample IM maximizer because the left-hand side is 0 there and c >= 0.
    This is not a proven finite-sample coverage statement, and it is not
    an accuracy-risk guarantee.
    """
    u_hat = np.asarray(u_hat, dtype=np.float64)
    e = np.asarray(u_boot, dtype=np.float64) - u_hat[None, :]
    t = e.max(axis=1) - e.min(axis=1)
    c = float(np.quantile(t, 1.0 - alpha))
    members = np.flatnonzero((float(u_hat.max()) - u_hat) <= c + 1e-15)
    return {"members": members.astype(np.int64), "critical_value": c, "t": t}


def set_median(members: np.ndarray) -> int:
    """Middle position of the already-sorted set; even size -> larger middle."""
    n = len(members)
    if n == 0:
        raise RuntimeError("empty admissible set")
    return int(members[n // 2])


def one_se_set(u_hat: np.ndarray, u_boot: np.ndarray) -> dict:
    """1-SE-style set using bootstrap SD at the IM-argmax candidate.

    Adaptation of the 1-SE idea, not a formal statistical confidence set.
    """
    b_hat = argmax_largest_b(u_hat)
    sd = float(np.std(u_boot[:, b_hat], ddof=1)) if u_boot.shape[0] > 1 else 0.0
    members = np.flatnonzero((float(u_hat.max()) - u_hat) <= sd + 1e-15)
    return {"members": members.astype(np.int64), "critical_value": sd}


def minimax_choice(members: np.ndarray, cost_ratio: float) -> int:
    """Finite asymmetric minimax over j in S, working hypothesis only.

    L(a, j) = cT (j-a)_+ + cR (a-j)_+ with cR=1, cT=cost_ratio.
    For j restricted to S this is max(cT (max S - a), cR (a - min S)).
    Ties take the largest a. Not an accuracy-risk guarantee.
    """
    members = np.asarray(members, dtype=np.int64)
    lo = int(members[0])
    hi = int(members[-1])
    c_t = float(cost_ratio)
    c_r = 1.0
    best_a = int(members[-1])
    best_risk = np.inf
    for a in members:
        a = int(a)
        risk = max(c_t * (hi - a), c_r * (a - lo))
        if risk < best_risk - 1e-15 or (abs(risk - best_risk) <= 1e-15 and a > best_a):
            best_risk = risk
            best_a = a
    return best_a


def bootstrap_vote(u_boot: np.ndarray) -> tuple[int, np.ndarray]:
    """Mode of bootstrap argmax; numerical ties take the largest b."""
    votes = np.array([argmax_largest_b(u_boot[r]) for r in range(u_boot.shape[0])],
                     dtype=np.int64)
    counts = np.bincount(votes, minlength=u_boot.shape[1])
    choice = int(np.flatnonzero(counts == counts.max())[-1])
    return choice, counts


def format_members(members: np.ndarray) -> str:
    return ";".join(str(int(x)) for x in members)


# ---------------------------------------------------------------------------
# Full selector battery on one CAL matrix (label-free)
# ---------------------------------------------------------------------------

def weighted_cdf_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Generalized inverse of the weighted empirical CDF: inf{x: F(x) >= q}.

    Ties keep the original relative order via stable argsort. This is a
    step-function quantile, not NumPy's linear interpolation.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    if values.shape != weights.shape:
        raise RuntimeError("values/weights length mismatch")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise RuntimeError("weights must be finite and non-negative")
    wsum = float(weights.sum())
    if wsum <= 0:
        raise RuntimeError("weights must sum to a positive value")
    q = float(q)
    if not (0.0 < q <= 1.0):
        raise RuntimeError(f"q must be in (0, 1], got {q}")
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    cdf = np.cumsum(weights[order]) / wsum
    idx = int(np.searchsorted(cdf, q, side="left"))
    if idx >= v.size:
        idx = v.size - 1
    return float(v[idx])


def select_from_cal(p_cal: np.ndarray, B: int, seed_tags: list[int],
                    alphas=None, mechanism: bool = True,
                    order_ablation: bool = True, unpaired: bool = False,
                    setrandom_seeds=None, minimax_ratios=None) -> dict:
    """Run IM + set actions sharing one paired bootstrap.

    seed_tags identify the CAL unit (grid, ci, seed, fold, N). They must
    not include labels or evaluation statistics. Bootstrap index streams
    depend only on seed_tags and B.

    The selector consumes a probability tensor. Rows are re-normalized at
    entry; already-simplex inputs are unchanged up to float64 roundoff.
    Labels, true fc, family, and evaluation statistics are not accepted.
    """
    if alphas is None:
        alphas = ALPHAS
    if setrandom_seeds is None:
        setrandom_seeds = SETRANDOM_DECISION_SEEDS
    if minimax_ratios is None:
        minimax_ratios = MINIMAX_RATIOS
    p_cal = normalize_probabilities(p_cal)
    k = p_cal.shape[0]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([CAL_BOOT_SS] + seed_tags)))
    u_boot, u_hat = paired_bootstrap_im(p_cal, B, rng, paired=True)
    b_im = argmax_largest_b(u_hat)
    out = {
        "u_hat": u_hat,
        "b_IM": b_im,
        "sets": {},
        "choices": {"IM": b_im},
        "setrandom_choices": {},
        "minimax": {},
    }
    for a in alphas:
        s = admissible_set(u_hat, u_boot, a)
        out["sets"][a] = s
        members = s["members"]
        if len(members) == 0:
            raise RuntimeError("empty admissible set")
        if b_im not in set(int(x) for x in members):
            raise RuntimeError("admissible set does not contain IM maximizer")
        if np.any((members < 0) | (members >= k)):
            raise RuntimeError("illegal set member")
        tag = "" if abs(a - ALPHA_PRIMARY) < 1e-12 else f"_a{a:.2f}"
        out["choices"][f"SetMin{tag}"] = int(members[0])
        out["choices"][f"SetMax{tag}"] = int(members[-1])
        out["choices"][f"SetMedian{tag}"] = set_median(members)
        rchoices = np.empty(setrandom_seeds, dtype=np.int64)
        for i in range(setrandom_seeds):
            rr = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([SETRANDOM_SS] + seed_tags + [int(round(a * 100)), i])))
            rchoices[i] = int(members[rr.integers(0, len(members))])
        out["setrandom_choices"][a] = rchoices
        out["choices"][f"SetRandom{tag}"] = rchoices
        if minimax_ratios:
            for ratio in minimax_ratios:
                name = f"Minimax_r{int(ratio)}{tag}"
                out["choices"][name] = minimax_choice(members, ratio)
                out["minimax"][(a, ratio)] = out["choices"][name]

    if mechanism:
        vote, counts = bootstrap_vote(u_boot)
        out["choices"]["BootstrapVote"] = vote
        out["vote_counts"] = counts
        s1 = one_se_set(u_hat, u_boot)
        out["choices"]["OneSEMax"] = int(s1["members"][-1])
        out["sets_1se"] = s1

    if order_ablation:
        perm_rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([PERM_SS] + seed_tags)))
        perm = perm_rng.permutation(k)
        u_hat_p = u_hat[perm]
        u_boot_p = u_boot[:, perm]
        s_p = admissible_set(u_hat_p, u_boot_p, ALPHA_PRIMARY)
        b_im_p = argmax_largest_b(u_hat_p)
        out["permutation"] = perm
        out["choices"]["PermutedIM"] = int(perm[b_im_p])
        out["choices"]["PermutedSetMax"] = int(perm[s_p["members"][-1]])
        out["choices"]["PermutedSetMin"] = int(perm[s_p["members"][0]])

    if unpaired:
        rng_u = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([UNPAIRED_SS] + seed_tags)))
        u_boot_u, _ = paired_bootstrap_im(p_cal, B, rng_u, paired=False)
        s_u = admissible_set(u_hat, u_boot_u, ALPHA_PRIMARY)
        out["unpaired_set"] = s_u
        out["choices"]["UnpairedSetMax"] = int(s_u["members"][-1])
        out["choices"]["UnpairedSetMin"] = int(s_u["members"][0])
        out["choices"]["UnpairedSetMedian"] = set_median(s_u["members"])
    return out


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def _meta_scalar(z, key, default=None):
    if key not in z.files:
        return default
    v = z[key]
    return v.item() if getattr(v, "shape", None) == () else v


def gridc_cond_id(condition: str) -> tuple[str, dict]:
    """Reconstruct the historical Grid C cond_id from the NPZ condition string.

    analyze_gridc_formal defaults order_or_param to the full condition
    string and fc_or_slope to the fc token if present, else ''.
    """
    sm = _SNR_RE.search(condition)
    snr = int(sm.group(1)) if sm else None
    fm = _FC_RE.search(condition)
    fc = int(fm.group(1)) if fm else None
    fc_or_slope = str(fc) if fc is not None else ""
    cid = f"{condition}|{fc_or_slope}|snr{snr}"
    return cid, {"fc_hz": fc, "snr_db": snr, "condition_str": condition}


def load_npz(path: Path, normalize: bool = True) -> dict:
    with np.load(path, allow_pickle=False) as z:
        raw = np.asarray(z["probabilities"])
        p = np.asarray(raw, dtype=np.float64)
        p, sums = validate_probability_tensor(p)
        if normalize:
            p = p / sums
        if "true_label" in z.files:
            labels = np.asarray(z["true_label"], dtype=np.int64)
            ids = np.asarray(z["clip_id"]).astype(str)
        else:
            labels = np.asarray(z["targets"], dtype=np.int64)
            ids = np.asarray(z["filenames"]).astype(str)
        boundaries = np.asarray(z["boundaries"], dtype=np.int64)
        meta = {}
        for k in z.files:
            if k in {"probabilities", "true_label", "targets", "clip_id",
                     "filenames", "boundaries"}:
                continue
            meta[k] = _meta_scalar(z, k)
    n_clips = int(p.shape[1])
    if labels.shape[0] != n_clips or ids.shape[0] != n_clips:
        raise RuntimeError(f"labels/IDs length mismatch in {path}")
    if len(np.unique(ids)) != n_clips:
        raise RuntimeError(f"duplicate clip IDs in {path}")
    if p.shape[0] != len(boundaries):
        raise RuntimeError(f"boundary/probability mismatch in {path}")
    if len(boundaries) == 0:
        raise RuntimeError(f"empty boundary list in {path}")
    if int(boundaries[0]) != 0:
        raise RuntimeError(
            f"boundaries in {path} do not start at 0; array index is not b"
        )
    if len(boundaries) > 1 and not (np.diff(boundaries) == 1).all():
        raise RuntimeError(f"non-contiguous boundaries in {path}")
    if np.any(p < 0):
        raise RuntimeError(f"negative probability in {path}")
    preds = np.argmax(p, axis=2).astype(np.int16)
    return {
        "p": p, "labels": labels, "ids": ids, "boundaries": boundaries,
        "preds": preds, "meta": meta, "path": str(path),
        "raw_dtype": str(raw.dtype),
    }


def load_grid_a(cache_dir: Path, normalize: bool = True) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for f in sorted(Path(cache_dir).glob("*.npz")):
        d = load_npz(f, normalize=normalize)
        cond = str(d["meta"]["cond_id"])
        fold = int(d["meta"]["fold"])
        d["family"] = str(d["meta"]["family"])
        d["cond_id"] = cond
        d["condition"] = cond
        d["condition_str"] = cond
        d["fc_hz"] = int(d["meta"]["fc_or_slope"]) if str(d["meta"].get("fc_or_slope", "")).isdigit() else None
        d["snr_db"] = int(d["meta"]["snr_db"]) if d["meta"].get("snr_db") is not None else None
        key = (cond, fold)
        if key in out:
            raise RuntimeError(f"duplicate Grid A condition/fold: {key}")
        out[key] = d
    return out


def load_grid_c(cache_root: Path, normalize: bool = True) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for f in sorted(Path(cache_root).glob("**/*.npz")):
        if "quarantine" in str(f).replace("\\", "/"):
            continue
        d = load_npz(f, normalize=normalize)
        condition = str(d["meta"]["condition"])
        fold = int(d["meta"]["fold"])
        cid, extra = gridc_cond_id(condition)
        d["family"] = str(d["meta"]["family"])
        d["cond_id"] = cid
        d["condition"] = cid
        d["condition_str"] = condition
        d["fc_hz"] = extra["fc_hz"]
        d["snr_db"] = extra["snr_db"]
        key = (cid, fold)
        if key in out:
            raise RuntimeError(f"duplicate Grid C condition/fold: {key}")
        out[key] = d
    return out


def condition_order(grid: str, caches: dict, protocol: str = "20_380") -> list[str]:
    """Legacy deterministic condition ordering used by the matching split RNG.

    Grid C 20/380: sorted by (family, NPZ condition string).
    Grid A 20/380 (exp13): sorted by (family, cond_id).
    Grid A budget (exp21): sorted cond_id strings only.
    """
    seen: dict[str, dict] = {}
    for (cid, fold), d in caches.items():
        if cid not in seen:
            seen[cid] = d
    if grid == "C":
        return [cid for cid, _ in sorted(
            seen.items(), key=lambda kv: (str(kv[1]["family"]), str(kv[1]["condition_str"]))
        )]
    if protocol == "80_320":
        return sorted(seen.keys())
    return [kv[0] for kv in sorted(
        seen.items(), key=lambda kv: (str(kv[1]["family"]), str(kv[0]))
    )]


def split_permutation(protocol: str, ci: int, seed: int, fold: int,
                      n_clips: int = 400) -> np.ndarray:
    if protocol == "80_320":
        ss = SPLIT_SS_BUDGET
    elif protocol == "20_380":
        ss = SPLIT_SS_GRID
    else:
        raise RuntimeError(f"unknown split protocol: {protocol}")
    return np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([ss, ci, seed, fold]))).permutation(n_clips)


BUDGET_SAMPLING_LEGACY = "legacy_sorted_prefix"
BUDGET_SAMPLING_CORRECTED = "corrected_random_nested"
BUDGET_SAMPLING_MODES = (BUDGET_SAMPLING_LEGACY, BUDGET_SAMPLING_CORRECTED)


def cal_eval_indices(protocol: str, perm: np.ndarray, n_cal: int,
                     budget_sampling: str = BUDGET_SAMPLING_LEGACY
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted CAL / EVAL index arrays.

    Sorting happens after the subset is chosen. Nested N share one
    permutation: smaller N is a prefix of the 80-pool draw, EVAL is
    identical across N.

    20/380: 1 <= N <= 20; CAL = sort(perm[:N]); EVAL = sort(perm[20:]).
    80/320:
      * legacy_sorted_prefix (historical exp21/23 replay only):
            pool = sort(perm[:80]); CAL = pool[:N]
        This is not a uniform random N-subset of the pool.
      * corrected_random_nested (official budget protocol):
            pool_ordered = perm[:80]   # keep random order
            CAL = sort(pool_ordered[:N])
        N=80 coincides with legacy. EVAL = sort(perm[80:]) in both modes.
    """
    perm = np.asarray(perm)
    n_cal = int(n_cal)
    if protocol == "80_320":
        if budget_sampling not in BUDGET_SAMPLING_MODES:
            raise RuntimeError(f"unknown budget_sampling: {budget_sampling}")
        if not (1 <= n_cal <= 80):
            raise RuntimeError(f"80_320 requires 1<=N<=80, got {n_cal}")
        ev = np.sort(perm[80:])
        pool_ordered = np.asarray(perm[:80])
        if budget_sampling == BUDGET_SAMPLING_LEGACY:
            cal = np.sort(pool_ordered)[:n_cal]
        else:
            cal = np.sort(pool_ordered[:n_cal])
        if cal.size == 0:
            raise RuntimeError("empty CAL")
        return cal.astype(np.int64, copy=False), ev.astype(np.int64, copy=False)
    if protocol != "20_380":
        raise RuntimeError(f"unknown split protocol: {protocol}")
    if not (1 <= n_cal <= 20):
        raise RuntimeError(f"20_380 requires 1<=N<=20, got {n_cal}")
    cal = np.sort(perm[:n_cal])
    ev = np.sort(perm[20:])
    if cal.size == 0:
        raise RuntimeError("empty CAL")
    return cal.astype(np.int64, copy=False), ev.astype(np.int64, copy=False)


def offline_accuracy_geometry(curve: np.ndarray, b_cf: int | None = None,
                              deltas=(0.0, 0.5, 1.0), steps=(1, 2, 3),
                              atol: float = 1e-12) -> dict:
    """Label-assisted offline geometry of an accuracy curve.

    Not a selector input. Losses from Amax are nonnegative. The historical
    CF-centered signed change is retained under a separate name and may be
    negative. Near-optimal sets are not filled across gaps; min/max of a
    disconnected set is not treated as a platform.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 1 or curve.size == 0:
        raise RuntimeError(f"accuracy curve must be 1-D nonempty, got {curve.shape}")
    if not np.isfinite(curve).all():
        raise RuntimeError("non-finite accuracy curve")
    klen = int(curve.size)
    amax = float(curve.max())
    max_points = np.flatnonzero(np.isclose(curve, amax, atol=atol, rtol=0.0)).astype(np.int64)
    near_opt = []
    for delta in deltas:
        delta = float(delta)
        members = np.flatnonzero(curve >= amax - delta).astype(np.int64)
        runs = []
        if members.size:
            breaks = np.where(np.diff(members) > 1)[0]
            starts = np.concatenate([[0], breaks + 1])
            ends = np.concatenate([breaks + 1, [members.size]])
            for a, b in zip(starts, ends):
                run = members[a:b]
                runs.append({
                    "left": int(run[0]),
                    "right": int(run[-1]),
                    "size": int(run.size),
                    "members": [int(x) for x in run],
                })
        near_opt.append({
            "delta_pp": delta,
            "members": [int(x) for x in members],
            "size": int(members.size),
            "n_components": int(len(runs)),
            "contiguous": bool(len(runs) <= 1),
            "components": runs,
            "span_min": int(members.min()) if members.size else None,
            "span_max": int(members.max()) if members.size else None,
            "span_is_platform": bool(len(runs) == 1),
        })
    amax_steps = []
    for b_star in max_points:
        b_star = int(b_star)
        for k_step in steps:
            k_step = int(k_step)
            left_i = b_star - k_step
            right_i = b_star + k_step
            left = float(amax - curve[left_i]) if left_i >= 0 else float("nan")
            right = float(amax - curve[right_i]) if right_i < klen else float("nan")
            both = np.isfinite(left) and np.isfinite(right)
            if not both:
                ratio = float("nan")
                ratio_status = "missing_side"
            elif left == 0.0 and right == 0.0:
                ratio = float("nan")
                ratio_status = "both_zero"
            elif left == 0.0:
                ratio = float("nan")
                ratio_status = "denominator_zero"
            else:
                ratio = float(right / left)
                ratio_status = "defined"
            amax_steps.append({
                "center": "amax_point",
                "b_star": b_star,
                "step": k_step,
                "loss_left_pp": left,
                "loss_right_pp": right,
                "right_minus_left_pp": float(right - left) if both else float("nan"),
                "right_over_left": ratio,
                "ratio_status": ratio_status,
            })
    cf_signed = []
    if b_cf is not None:
        b_ref = int(b_cf)
        if not (0 <= b_ref < klen):
            raise RuntimeError(f"b_cf out of range: {b_ref}")
        for k_step in steps:
            k_step = int(k_step)
            left_i = b_ref - k_step
            right_i = b_ref + k_step
            left = float(curve[b_ref] - curve[left_i]) if left_i >= 0 else float("nan")
            right = float(curve[b_ref] - curve[right_i]) if right_i < klen else float("nan")
            both = np.isfinite(left) and np.isfinite(right)
            if not both:
                ratio = float("nan")
                ratio_status = "missing_side"
            elif left == 0.0 and right == 0.0:
                ratio = float("nan")
                ratio_status = "both_zero"
            elif left == 0.0:
                ratio = float("nan")
                ratio_status = "denominator_zero"
            else:
                ratio = float(right / left)
                ratio_status = "defined"
            cf_signed.append({
                "center": "cf_mode",
                "b_CF": b_ref,
                "step": k_step,
                "signed_acc_change_left_pp": left,
                "signed_acc_change_right_pp": right,
                "right_minus_left_pp": float(right - left) if both else float("nan"),
                "right_over_left": ratio,
                "ratio_status": ratio_status,
            })
    return {
        "amax": amax,
        "max_points": [int(x) for x in max_points],
        "n_max_points": int(len(max_points)),
        "amax_atol": float(atol),
        "near_opt": near_opt,
        "amax_step_losses": amax_steps,
        "cf_signed_acc_change": cf_signed,
    }


def list_npz_paths_grid_a(cache_dir: Path) -> dict[tuple[str, int], Path]:
    out = {}
    for f in sorted(Path(cache_dir).glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            out[(str(z["cond_id"]), int(z["fold"]))] = f
    return out


def list_npz_paths_grid_c(cache_root: Path) -> dict[tuple[str, int], Path]:
    out = {}
    for f in sorted(Path(cache_root).glob("**/*.npz")):
        if "quarantine" in str(f).replace("\\", "/"):
            continue
        with np.load(f, allow_pickle=False) as z:
            condition = str(z["condition"].item() if getattr(z["condition"], "shape", None) == () else z["condition"])
            fold = int(z["fold"].item() if getattr(z["fold"], "shape", None) == () else z["fold"])
            cid, _ = gridc_cond_id(condition)
            out[(cid, fold)] = f
    return out

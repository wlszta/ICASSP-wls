import numpy as np

from trainingless_adaptation.selection import argmax_largest_b, im_scores, select_from_cal


def test_im_scores_and_largest_boundary_tie_rule():
    p = np.full((3, 4, 2), 0.5, dtype=float)
    assert np.allclose(im_scores(p), 0.0)
    assert argmax_largest_b(np.array([1.0, 1.0, 0.5])) == 1


def test_selector_uses_unlabeled_probability_tensor():
    p = np.array([
        [[0.95, 0.05], [0.9, 0.1], [0.8, 0.2]],
        [[0.55, 0.45], [0.5, 0.5], [0.45, 0.55]],
    ])
    result = select_from_cal(p, B=8, seed_tags=[1, 2, 3])
    assert result["b_IM"] in (0, 1)
    assert result["choices"]["IM"] == result["b_IM"]

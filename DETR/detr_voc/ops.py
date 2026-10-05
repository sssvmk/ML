"""Box operations and the Hungarian assignment (scipy when it works, a pure numpy solver otherwise)."""
import numpy as np
import torch
from torchvision.ops import generalized_box_iou


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def giou_matrix(a_xyxy: torch.Tensor, b_xyxy: torch.Tensor) -> torch.Tensor:
    """Generalized IoU (paper eq. 10) between every box of a and every box of b."""
    return generalized_box_iou(a_xyxy, b_xyxy)


def hungarian_numpy(cost: np.ndarray):
    """Minimum-cost assignment for a rectangular cost matrix (O(n^2 m), potentials method). Returns (rows, cols) like scipy."""
    cost = np.asarray(cost, dtype=np.float64)
    transposed = cost.shape[0] > cost.shape[1]
    c = cost.T if transposed else cost
    n, m = c.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    inf = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=np.int64), np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(m + 1, inf), np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = c[i0 - 1, :] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            masked = np.where(free, minv[1:], inf)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows)
    return rows[order], cols[order]


_SCIPY = {}


def solve_assignment(cost: np.ndarray):
    """scipy's linear_sum_assignment if scipy imports cleanly (a cluster with a numpy/scipy mismatch raises on import),
    otherwise the numpy solver above. Same optimal cost either way."""
    if "fn" not in _SCIPY:
        try:
            from scipy.optimize import linear_sum_assignment
            _SCIPY["fn"] = linear_sum_assignment
        except Exception:  # noqa: BLE001 - broken scipy installs raise AttributeError, not ImportError
            _SCIPY["fn"] = hungarian_numpy
    return _SCIPY["fn"](cost)

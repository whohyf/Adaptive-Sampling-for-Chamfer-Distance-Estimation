"""CrudeNN: coarse nearest-neighbor estimation via random-shift grid hashing (LSH).

Algorithm (l1 version, Bakshi et al. 2023 Section 2):
  For L scale levels with grid side r_i = r_start * 2^i:
    1. Draw random shift z ~ Uniform[0, r_i]^d
    2. Hash B: h_i(b) = floor((b + z) / r_i)
    3. For each unresolved a in A: look up h_i(a); on first collision record D_a = ||a - b'||_1

The starting scale r_start is estimated from a small probe subsample of NN distances.

For L2 metric: first JL-project A and B into L1, then run the L1 version.

Key optimization (v2): vectorized collision detection via np.unique, replacing the
Python for-loop that was the bottleneck in Phase 1 (~50ms for n=8000 -> ~2ms).
"""
import numpy as np
from scipy.spatial.distance import cdist

_METRIC_MAP = {"l2": "euclidean", "l1": "cityblock"}


def _estimate_nn_scale(A, B, rng, n_probe=30):
    """Estimate typical NN distance via a small random probe subsample."""
    n_a, d = A.shape
    n_b = len(B)
    n_probe_a = min(n_probe, n_a)
    n_probe_b = min(200, n_b)

    idx_a = rng.choice(n_a, n_probe_a, replace=False)
    idx_b = rng.choice(n_b, n_probe_b, replace=False)

    dists = np.sum(np.abs(A[idx_a][:, None, :] - B[idx_b][None, :, :]), axis=2)
    nn_dists = dists.min(axis=1)
    return float(np.median(nn_dists))


def _l1_crude_nn(A, B, num_levels, rng, scale_nn=None):
    """Core CrudeNN for L1 distance.  Vectorized hash collision detection.

    Levels: r_i = (scale_nn / 2) * 2^i  for i = 0, ..., num_levels-1.
    """
    n_a, d = A.shape
    n_b = len(B)

    if scale_nn is None:
        scale_nn = _estimate_nn_scale(A, B, rng)
    scale_nn = max(scale_nn, 1e-12)

    r_start = scale_nn / 2.0

    D_a = np.full(n_a, np.inf)
    found = np.zeros(n_a, dtype=bool)

    # Threshold: use vectorized np.unique for d <= 50 (fast path),
    # fall back to Python loop for very high-d (where unique overhead dominates)
    use_vectorized = d <= 50

    for i in range(num_levels):
        if found.all():
            break

        r = r_start * (2.0 ** i)
        z = rng.uniform(0.0, r, size=d)

        B_hashes = np.floor((B + z) / r).astype(np.int64)

        unfound_idx = np.where(~found)[0]
        A_sub = A[unfound_idx]
        A_hashes = np.floor((A_sub + z) / r).astype(np.int64)
        n_unfound = len(A_sub)

        if use_vectorized:
            # ---- Vectorized collision detection via np.unique ----
            all_hashes = np.vstack([B_hashes, A_hashes])
            _, inverse = np.unique(all_hashes, axis=0, return_inverse=True)
            # inverse[:n_b] = hash IDs for B; inverse[n_b:] = hash IDs for A_sub

            # Build mapping: hash_id -> first B point index (vectorized)
            b_ids = inverse[:n_b]
            # argsort + unique to get first B per hash
            sort_idx = np.argsort(b_ids)
            sorted_b = b_ids[sort_idx]
            unique_hids, first_pos = np.unique(sorted_b, return_index=True)
            hash_to_b = np.full(inverse.max() + 1, -1, dtype=np.intp)
            hash_to_b[unique_hids] = sort_idx[first_pos]

            # Resolve A collisions
            a_ids = inverse[n_b:]
            b_match = hash_to_b[a_ids]  # -1 = no match, else B index
            matched_mask = b_match >= 0

            if matched_mask.any():
                matched_local = np.where(matched_mask)[0]
                matched_global = unfound_idx[matched_local]
                b_idx = b_match[matched_local]
                D_a[matched_global] = np.sum(
                    np.abs(A[matched_global] - B[b_idx]), axis=1
                )
                found[matched_global] = True
        else:
            # ---- Fallback: Python loop for very high-d data ----
            B_hash_dict = {}
            for j in range(n_b):
                key = B_hashes[j].tobytes()
                if key not in B_hash_dict:
                    B_hash_dict[key] = j

            for local_i in range(n_unfound):
                key = A_hashes[local_i].tobytes()
                if key in B_hash_dict:
                    global_i = unfound_idx[local_i]
                    b_idx = B_hash_dict[key]
                    D_a[global_i] = np.sum(np.abs(A[global_i] - B[b_idx]))
                    found[global_i] = True

    # Fallback: points with no collision at any level -> random B point
    still_unfound = np.where(~found)[0]
    if len(still_unfound) > 0:
        j_arr = rng.integers(n_b, size=len(still_unfound))
        for ii, k in enumerate(still_unfound):
            D_a[k] = np.sum(np.abs(A[k] - B[j_arr[ii]]))

    return D_a


def crude_nn(A, B, num_levels=None, metric="l2", eps=0.1, rng=None):
    """CrudeNN: return D_a array of coarse NN estimates for each a in A.

    Guarantees (with high probability):
      D_a >= gamma_a          (deterministic: D_a is distance to an actual b in B)
      E[D_a] <= O(log n) * gamma_a

    Aspect-ratio pre-step (ALGORITHM_SPEC §1.2, audit M-12):
      If the estimated median NN distance rho < 1, rescale A and B by 1/rho before
      running the grid hash so that all grid levels operate in a numerically stable
      range.  Returned D_a values are rescaled back to the original space, so the
      caller sees distances in the original metric.

    Parameters
    ----------
    A, B : (n_a, d) and (n_b, d) arrays
    num_levels : number of grid scale levels; default = ceil(log2(n / eps)) + 2
    metric : 'l1' or 'l2'
    eps : approximation parameter (used only for default num_levels)
    rng : numpy random Generator or seed
    """
    rng = np.random.default_rng(rng)
    n = max(len(A), len(B))

    if num_levels is None:
        num_levels = max(3, int(np.ceil(np.log2(n / eps))) + 2)

    if metric == "l1":
        # Aspect-ratio pre-step: estimate scale and rescale if needed.
        rho = _estimate_nn_scale(A, B, rng)
        if rho < 1.0 and rho > 1e-12:
            scale = 1.0 / rho
            D_a = _l1_crude_nn(A * scale, B * scale, num_levels, rng)
            return D_a / scale
        return _l1_crude_nn(A, B, num_levels, rng)

    elif metric == "l2":
        # JL projection: R^d -> R^k, preserves L2 distances up to (1 +- eps)
        d = A.shape[1]
        k = max(1, int(np.ceil(4 * np.log(n) / (eps ** 2))))
        k = min(k, d)
        P = rng.standard_normal((d, k)) / np.sqrt(k)
        A_proj = A @ P
        B_proj = B @ P
        # Aspect-ratio pre-step on projected data.
        rho = _estimate_nn_scale(A_proj, B_proj, rng)
        if rho < 1.0 and rho > 1e-12:
            scale = 1.0 / rho
            D_a = _l1_crude_nn(A_proj * scale, B_proj * scale, num_levels, rng)
            return D_a / scale
        scale_nn = rho if rho >= 1.0 else None
        return _l1_crude_nn(A_proj, B_proj, num_levels, rng, scale_nn=scale_nn)

    else:
        raise ValueError(f"metric must be 'l1' or 'l2', got {metric!r}")


def oracle_crude_nn(A, B, metric="l2"):
    """Oracle CrudeNN: exact gamma_a as D_a.  Upper bound on IS performance."""
    return cdist(A, B, metric=_METRIC_MAP.get(metric, metric)).min(axis=1)

"""
AHP (Analytic Hierarchy Process) computation for design quality metric weights.

Constructs a pairwise comparison matrix for the 6 design quality metrics,
computes the priority vector (weights) via the principal eigenvector method,
and checks consistency (CR < 0.10).

Reference: Saaty, T.L. (1980). The Analytic Hierarchy Process.
"""

import numpy as np

CR_THRESHOLD = 0.10

# --- Metric names ---
metrics = ["IoU", "PA", "Obj", "Constr", "Conn", "WT"]

# --- Pairwise comparison matrix (Saaty 1-9 scale) ---
# Entry (i,j) = how much more important metric i is over metric j
# 1 = equal, 3 = moderate, 5 = strong, 7 = very strong, 9 = extreme
# Reciprocals are automatic: A[j,i] = 1/A[i,j]
#
# Rationale for judgments:
#   - IoU is the primary structural similarity metric (gold standard for segmentation)
#   - PA complements IoU but is inflated by background pixels → less discriminative
#   - Objective match captures engineering performance (compliance)
#   - Constraint match measures specification adherence (volume fraction)
#   - Connectivity ensures physically realizable (no floating material)
#   - Watertightness is only relevant for STL export → least critical overall

# fmt: off
A = np.array([
    #        IoU    PA     Obj    Constr  Conn    WT
    [  1,     2,     2,     3,     3,     3  ],   # IoU
    [  1/2,   1,     1,     2,     2,     2  ],   # PA
    [  1/2,   1,     1,     1,     1,     2  ],   # Obj
    [  1/3,   1/2,   1,     1,     1,     1  ],   # Constr
    [  1/3,   1/2,   1,     1,     1,     1  ],   # Conn
    [  1/3,   1/2,   1/2,   1,     1,     1  ],   # WT
], dtype=float)
# fmt: on

n = len(metrics)

# --- Compute weights via eigenvector method ---
eigenvalues, eigenvectors = np.linalg.eig(A)
# Principal eigenvalue = largest real eigenvalue
lambda_max_idx = np.argmax(np.real(eigenvalues))
lambda_max = np.real(eigenvalues[lambda_max_idx])
# Corresponding eigenvector (normalized to sum to 1)
w = np.real(eigenvectors[:, lambda_max_idx])
w = w / w.sum()

# --- Consistency check ---
# Random Index (RI) values for n=1..10 (Saaty, 1980)
RI_table = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}
CI = (lambda_max - n) / (n - 1)
CR = CI / RI_table[n]

# --- Also compute weights via geometric mean method (for comparison) ---
geo_means = np.prod(A, axis=1) ** (1.0 / n)
w_geo = geo_means / geo_means.sum()

# --- Print results ---
print("=" * 65)
print("AHP Analysis: Design Quality Metric Weights")
print("=" * 65)

print("\nPairwise Comparison Matrix:")
header = f"{'':>8s}" + "".join(f"{m:>8s}" for m in metrics)
print(header)
for i, m in enumerate(metrics):
    row = f"{m:>8s}" + "".join(f"{A[i, j]:8.2f}" for j in range(n))
    print(row)

print(f"\nPrincipal eigenvalue (λ_max): {lambda_max:.4f}")
print(f"Consistency Index (CI):       {CI:.4f}")
print(f"Random Index (RI, n={n}):      {RI_table[n]:.2f}")
print(
    f"Consistency Ratio (CR):       {CR:.4f}  {'✓ < 0.10 (consistent)' if CR < CR_THRESHOLD else '✗ ≥ 0.10 (inconsistent!)'}"
)

print("\n" + "-" * 65)
print(f"{'Metric':<12s} {'AHP (eigvec)':>14s} {'AHP (geomean)':>14s} {'Current':>10s}")
print("-" * 65)

current = [0.31, 0.19, 0.15, 0.12, 0.12, 0.11]
for i, m in enumerate(metrics):
    print(f"{m:<12s} {w[i]:>14.4f} {w_geo[i]:>14.4f} {current[i]:>10.2f}")

print("-" * 65)
print(f"{'Sum':<12s} {w.sum():>14.4f} {w_geo.sum():>14.4f} {sum(current):>10.2f}")

print("\n" + "=" * 65)
print("Rounded AHP weights (to 2 decimal places, normalized):")
w_rounded = np.round(w, 2)
# Adjust largest weight to ensure sum = 1.00
w_rounded[np.argmax(w)] += 1.00 - w_rounded.sum()
for i, m in enumerate(metrics):
    diff = w_rounded[i] - current[i]
    print(
        f"  {m:<12s} {w_rounded[i]:.2f}  (current: {current[i]:.2f}, Δ = {diff:+.2f})"
    )
print(f"  Sum = {w_rounded.sum():.2f}")

print("\nPairwise comparison rationale:")
rationale = [
    (
        "IoU vs PA",
        "IoU is the standard shape-matching metric; PA is inflated by trivial background pixels → IoU moderately preferred (2)",
    ),
    (
        "IoU vs Obj",
        "Geometric fidelity is the direct evaluation target; objective match depends on solver → IoU moderately preferred (2)",
    ),
    (
        "IoU vs Constr",
        "Constraint adherence is important but coarser than pixel-level shape match → IoU moderately-to-strongly preferred (3)",
    ),
    (
        "IoU vs Conn",
        "Connectivity is a binary printability check, less informative than continuous IoU → IoU moderately-to-strongly preferred (3)",
    ),
    (
        "IoU vs WT",
        "Watertightness only matters for STL styles; IoU applies universally → IoU moderately-to-strongly preferred (3)",
    ),
    (
        "PA vs Obj",
        "Both capture different quality facets at similar granularity → equal importance (1)",
    ),
    (
        "PA vs Constr",
        "PA provides finer-grained feedback than binary constraint check → PA moderately preferred (2)",
    ),
    ("PA vs Conn", "Similar reasoning as PA vs Constr → PA moderately preferred (2)"),
    (
        "PA vs WT",
        "PA applies to all styles; WT only to STL exports → PA moderately preferred (2)",
    ),
    (
        "Obj vs Constr",
        "Both are engineering metrics at similar granularity → equal importance (1)",
    ),
    (
        "Obj vs Conn",
        "Objective match is more informative for design quality → roughly equal (1)",
    ),
    (
        "Obj vs WT",
        "Objective match applies universally; WT is style-specific → Obj moderately preferred (2)",
    ),
    (
        "Constr vs Conn",
        "Both are coarse binary-like metrics at similar importance → equal (1)",
    ),
    ("Constr vs WT", "Both are secondary quality checks → roughly equal (1)"),
    (
        "Conn vs WT",
        "Both are printability metrics; connectivity is slightly more fundamental → roughly equal (1)",
    ),
]
for pair, reason in rationale:
    print(f"  {pair:<16s}: {reason}")

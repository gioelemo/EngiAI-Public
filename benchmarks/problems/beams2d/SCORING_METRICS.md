# Beams2D Scoring Metrics

This document describes all the metrics used to evaluate agent performance on beam topology optimization tasks.

## Overall Score

The overall score is a weighted composite of multiple metrics that evaluate both design quality and structural performance:

```
Overall Score = 0.40 × IoU
              + 0.25 × Pixel Accuracy
              + 0.15 × (1 - min(Volume Fraction Error × 2, 1))
              + 0.20 × Compliance Score
```

**Score Range**: 0.0 to 1.0 (higher is better)

**Weight Distribution**:
- **40%** - IoU (topology shape match)
- **25%** - Pixel Accuracy (fine-grained match)
- **15%** - Volume Fraction Error (constraint adherence)
- **20%** - Compliance Score (structural performance)

**Note**: The weights are always consistent regardless of whether compliance data is available. If compliance data is missing, the compliance component contributes 0, properly penalizing agents that fail to extract or compute compliance values.

---

## Design Quality Metrics

These metrics evaluate how well the agent's design matches the ground truth optimal design.

### 1. IoU (Intersection over Union)

**Description**: Measures the overlap between the agent's design and the ground truth design at the topology level.

**Calculation**:
```python
agent_binary = (design_array > 0.5).astype(int)
gt_binary = (ground_truth > 0.5).astype(int)

intersection = np.logical_and(agent_binary, gt_binary).sum()
union = np.logical_or(agent_binary, gt_binary).sum()
iou = intersection / union
```

**Range**: 0.0 to 1.0
- **1.0** = Perfect overlap
- **0.0** = No overlap

**Interpretation**: Higher IoU indicates better topology matching. This is the most important design metric, weighted at 40%.

---

### 2. Pixel Accuracy

**Description**: Measures the percentage of pixels that match between the agent's binary design and the ground truth binary design.

**Calculation**:
```python
agent_binary = (design_array > 0.5).astype(int)
gt_binary = (ground_truth > 0.5).astype(int)
pixel_accuracy = np.mean(agent_binary == gt_binary)
```

**Range**: 0.0 to 1.0
- **1.0** = All pixels match
- **0.5** = Half of pixels match
- **0.0** = No pixels match

**Interpretation**: Pixel accuracy provides a fine-grained view of design quality, complementing IoU. Weighted at 25%.

---

### 3. MSE (Mean Squared Error)

**Description**: Measures the average squared difference between design density values (not used in overall score, but reported for analysis).

**Calculation**:
```python
mse = np.mean((design_array - ground_truth) ** 2)
```

**Range**: 0.0 to 1.0 (lower is better)
- **0.0** = Perfect match
- **Higher values** = Greater deviation

**Interpretation**: MSE captures density-level differences, useful for understanding how closely the agent matched the continuous density distribution.

---

## Constraint Adherence Metrics

### 4. Volume Fraction Error

**Description**: Measures how well the agent adhered to the volume fraction constraint.

**Calculation**:
```python
agent_volfrac = np.mean(design_array)
gt_volfrac = np.mean(ground_truth)
volfrac_error = abs(agent_volfrac - gt_volfrac)
```

**Score Contribution**:
```python
volfrac_score = 1.0 - min(volfrac_error × 2, 1.0)
```

**Range**:
- **volfrac_error**: 0.0 to 1.0 (lower is better)
- **volfrac_score**: 0.0 to 1.0 (higher is better)

**Interpretation**:
- Error < 0.5 gets partial credit
- Error = 0 gets full credit (15% of overall score)
- Error ≥ 0.5 gets no credit

**Reported Metrics**:
- `volfrac_error`: Absolute difference in volume fractions
- `agent_volfrac`: Agent's material volume fraction
- `gt_volfrac`: Ground truth material volume fraction

---

## Structural Performance Metrics

### 5. Compliance Score

**Description**: Evaluates how close the agent's structural compliance is to the target optimal compliance.

**What is Compliance?**: In structural mechanics, compliance measures structural flexibility (lower is better). It's the inverse of stiffness and indicates how much a structure deforms under load.

**Calculation**:
```python
compliance_relative_error = abs(agent_compliance - target_compliance) / target_compliance
compliance_score = max(0.0, 1.0 - compliance_relative_error / 0.2)
```

**Range**: 0.0 to 1.0
- **1.0** = Agent compliance matches target (error = 0%)
- **0.8** = 20% relative error
- **0.0** = ≥20% relative error

**Interpretation**:
- The score linearly decreases as relative error increases
- Full credit within 0% error
- Zero credit at 20% error or higher
- Weighted at 20% of overall score

**Failure Cases** (score = 0.0):
- Compliance data not found in agent messages
- Agent didn't run optimization
- Target compliance not available

**Reported Metrics**:
- `compliance_score`: The normalized score (0-1)
- `agent_compliance`: Raw compliance value from agent's optimization
- `target_compliance`: Ground truth optimal compliance
- `compliance_relative_error`: `|agent - target| / target`

---

## Special Cases and Edge Cases

### No Design Found
```python
{
    "score": 0.0,
    "design_found": False,
    "reason": "No design array found in tool results or cache"
}
```

### Shape Mismatch
```python
{
    "score": 0.0,
    "design_found": True,
    "reason": "Shape mismatch: agent=(60, 20), ground_truth=(60, 20)",
    "agent_shape": "(60, 20)",
    "gt_shape": "(60, 20)"
}
```

### Missing Compliance Data
When compliance data cannot be extracted:
- `compliance_score = 0.0`
- `agent_compliance = None`
- `target_compliance = None`
- `compliance_relative_error = None`
- Overall score can still reach 0.8 (80%) from design metrics alone

---

## Scoring Philosophy

### Consistent Weights
All evaluations use the same weight distribution regardless of data availability. This ensures:
- **Comparability**: Scores across different runs are directly comparable
- **Fairness**: Missing data properly penalizes the agent (contributes 0)
- **Transparency**: Clear understanding of what each score means

### Balanced Evaluation
The scoring system balances:
- **Topology quality** (40% IoU + 25% pixel accuracy = 65%)
- **Constraint adherence** (15% volume fraction)
- **Structural performance** (20% compliance)

This reflects the importance of both matching the optimal design shape and achieving good structural performance.

### Partial Credit
Agents receive partial credit for:
- Good designs with missing compliance (up to 80%)
- Approximate volume fraction matches (scaled linearly)
- Close compliance values (scaled linearly up to 20% error)

---

## Example Scenarios

### Perfect Score (1.0)
```python
{
    "score": 1.0,
    "iou": 1.0,                      # Perfect topology match
    "pixel_accuracy": 1.0,           # All pixels match
    "volfrac_error": 0.0,            # Perfect constraint adherence
    "compliance_score": 1.0,         # Optimal structural performance
    "agent_compliance": 150.0,
    "target_compliance": 150.0
}
```

### Good Design, Missing Compliance (0.8)
```python
{
    "score": 0.8,
    "iou": 1.0,
    "pixel_accuracy": 1.0,
    "volfrac_error": 0.0,
    "compliance_score": 0.0,         # Missing data
    "agent_compliance": None,
    "target_compliance": 150.0
}
```

### Partial Match (0.5)
```python
{
    "score": 0.5,
    "iou": 0.6,                      # 60% topology match
    "pixel_accuracy": 0.75,          # 75% pixels match
    "volfrac_error": 0.1,            # 10% volume error
    "compliance_score": 0.5,         # 10% compliance error
    "agent_compliance": 165.0,
    "target_compliance": 150.0,
    "compliance_relative_error": 0.1
}
```

### Poor Performance (0.2)
```python
{
    "score": 0.2,
    "iou": 0.3,                      # Poor topology match
    "pixel_accuracy": 0.6,           # Many mismatches
    "volfrac_error": 0.4,            # 40% volume error
    "compliance_score": 0.0,         # >20% compliance error
    "agent_compliance": 200.0,
    "target_compliance": 150.0,
    "compliance_relative_error": 0.33
}
```

---

## Usage in Weave Evaluations

All metrics are automatically computed by the `score_design_match()` scorer function and logged to Weights & Biases via Weave:

```python
@weave.op()
def score_design_match(
    output: dict,
    metadata: dict[str, Any],
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Returns all metrics described above
    pass
```

The results are tracked in Weave for:
- Individual example evaluation
- Aggregate statistics across the dataset
- Comparison between different agent configurations
- Historical tracking of improvements

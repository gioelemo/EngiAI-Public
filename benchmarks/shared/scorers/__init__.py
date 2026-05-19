"""Scorers for engineering agent evaluations.

This module provides various scorers for evaluating agent performance:
- tool_use: Measures tool call efficiency and sequence correctness
- task_completion: Checks if required tools were called successfully
- output_quality: Evaluates design quality metrics
- hpc_workflow: Scores HPC training workflow completion

Note: Global metrics (MMD, DPP, IOG/COG/FOG, RVC) are now computed offline
by compute_global_metrics.py using the extracted design data.
"""

from benchmarks.shared.scorers.hpc_workflow_scorer import (
    score_hpc_workflow,
)
from benchmarks.shared.scorers.output_quality_scorer import (
    score_output_quality,
)
from benchmarks.shared.scorers.rag_scorer import (
    score_rag_evaluation,
)
from benchmarks.shared.scorers.task_completion_scorer import (
    score_task_completion,
)
from benchmarks.shared.scorers.tool_use_scorer import (
    score_tool_use,
)

__all__ = [
    "score_hpc_workflow",
    "score_output_quality",
    "score_rag_evaluation",
    "score_task_completion",
    "score_tool_use",
]

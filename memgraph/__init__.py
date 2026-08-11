"""memgraph — memory-operation relationship graphs from gate trajectories.

Extracts memory-related actions from `trajectory.jsonl` files and links them
across trajectories: which (trajectory, step) created each memory, and which
(trajectory, step) later recalled it.

Memory actions come from the schema-v6 `memory_*` annotation events recorded
by the traj-recorder gate: lanes from `memory_role_bind`, extractions from
generation operations, recalls from `memory_delivery` — explicit signals,
never label conventions, prompt prefixes, or payload guessing (see
`eval/memgraph/annotations.py`).
"""

from eval.memgraph.model import Edge, MemoryOp, MemoryUnit, Trajectory
from eval.memgraph.parse_trajectory import parse_trajectory
from eval.memgraph.graph import build_graph, discover_trajectories

__all__ = [
    "Edge",
    "MemoryOp",
    "MemoryUnit",
    "Trajectory",
    "parse_trajectory",
    "build_graph",
    "discover_trajectories",
]

__version__ = "0.1.0"

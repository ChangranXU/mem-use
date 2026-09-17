"""memgraph — memory-operation relationship graphs from gate trajectories.

Extracts memory-related actions from `trajectory.jsonl` files and links them
across trajectories: which (trajectory, step) created each memory, and which
(trajectory, step) later recalled it.

Memory actions come from the schema-v9 `memory_*` annotation events recorded
by the traj-recorder gate: lanes from `memory_role_bind`, extractions from
generation operations, recalls from `memory_delivery` — explicit signals,
never label conventions, prompt prefixes, or payload guessing (see
`memgraph/annotations.py`).
"""

from __future__ import annotations


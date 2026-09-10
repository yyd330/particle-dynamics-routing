# Particle Dynamics Routing (PDR)

Python implementation of **particle dynamics routing**: packets are treated as
particles launched under a gravity-like force field, and each network node
forwards a particle toward the neighbor whose link angle best matches the
particle's instantaneous trajectory. The approach is described in:

> Biswas, S., Yang, Y., Bhuyan, A. K., Dutta, H., & Datta, S. (2026).
> *Leveraging particle dynamics in force-fields for network packet routing.*
> PLOS ONE, 21(8), e0357202. https://doi.org/10.1371/journal.pone.0357202

## How it works

- **Topology** — A planar mesh of nodes is generated on a 10×10 plane: nodes
  are placed in a uniform grid with per-cell jitter, then linked by a
  crossing-free k-nearest-neighbor pass with minimum-degree and
  average-degree adjustment, followed by a connectivity check.
  (`planar_topology_implementation.py`)
- **Trajectory physics** — `trajectory_algorithm` integrates the particle's
  motion under uniform gravity hop by hop (iterative, cycle-guarded, capped at
  `MAX_HOPS`) and routes it to the neighbor whose link angle best matches its
  direction of travel.
- **Trajectory search** — `trajectory_search` finds the launch speeds that
  make a given node-to-node route reachable under a given gravity vector
  (bounded scan — safe, no float64 overflow).
- **Reachability & heat maps** — `search_reachability` sweeps launch
  parameters from a node to determine which nodes are physically reachable;
  `heat_map` aggregates those results per grid region and renders a
  reachability heat map.
- **Reverse trajectories** — `reverse_trajectory_algorithm` reconstructs the
  backward paths from the terminal records of two forward trajectories.
- **Shortest path** — `dijkstra_sp` computes hop-optimal all-pairs routes for
  comparison against PDR routes.

## Files

| File | Purpose |
|------|---------|
| `general_topology_implementation.py` | `NODE` and `TOPOLOGY` data structures (shared). |
| `planar_topology_implementation.py` | Planar-mesh topology generation + helper predicates. |
| `gravitational_algorithms.py` | PDR trajectory physics, reachability/heat maps, Dijkstra. Runnable CLI (`main()`). |
| `general_test.py` | Smoke test: builds the paper's 200-node topology, exercises every routing primitive. |
| `tests/` | pytest suite (30 tests: data structures, topology invariants, routing, plotting). |

## Setup

Python 3.10+ (3.11 recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Smoke test (topology stats + one of each routing primitive, ~20 s)
python3 general_test.py
python3 general_test.py --full        # larger reachability sweep

# Full pipeline (topology + reachability sweep + heat map + trajectory
# demos + reverse-trajectory demo). Plots are saved to ./output/ by default.
python3 gravitational_algorithms.py

# Fast demo scale (~60 s on 200 nodes) vs. full paper scale (~3 CPU-hours):
python3 gravitational_algorithms.py --scale fast --out ./output
python3 gravitational_algorithms.py --scale full --out ./output

# Single experiment only:
python3 gravitational_algorithms.py --experiment reachability
python3 gravitational_algorithms.py --experiment heatmap
python3 gravitational_algorithms.py --experiment trajectories
python3 gravitational_algorithms.py --experiment reverse

# Topology knobs:
python3 gravitational_algorithms.py --nodes 500 --grid-size 20 --seed 7
```

All plotting is headless-safe (Agg backend + `savefig`), so the pipeline runs
on machines without a display and in CI.

## Tests

```bash
python3 -m pytest tests/ -v
```

CI (GitHub Actions) runs the same suite on every push/PR:
`.github/workflows/ci.yml`.

## Note on trajectory semantics

Per the paper's convention, a node's `neighbors` list contains its own id at
index 0 (so `NODE.degree() == len(neighbors) - 1`), and the trajectory output
ends with a terminal `[node_id, vx, vy]` record. Sweep code therefore
collects node ids only from string entries.

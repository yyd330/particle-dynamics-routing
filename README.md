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
- **Trajectory search** — `trajectory_search` enumerates the launch speeds
  that make a given outgoing link produce distinct routes: an exponential
  phase establishes the speed interval (paper Sec. 10.1), then a binary
  search inside it records every trajectory change (bounded, no float64
  overflow).
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
| `tests/` | pytest suite (33 tests: data structures, topology invariants, routing, plotting). |

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
# demos + reverse-trajectory demo). Use --out DIR to save plots (PNGs).
python3 gravitational_algorithms.py --out ./output

# Fast demo scale (~60 s on 200 nodes, 12 workers) vs. full paper scale:
python3 gravitational_algorithms.py --scale fast --out ./output
python3 gravitational_algorithms.py --scale full --out ./output
python3 gravitational_algorithms.py --scale full --nodes 200 --workers 12

# Single experiment only:
python3 gravitational_algorithms.py --experiment reachability
python3 gravitational_algorithms.py --experiment heatmap
python3 gravitational_algorithms.py --experiment trajectories
python3 gravitational_algorithms.py --experiment reverse
python3 gravitational_algorithms.py --experiment dijkstra

# Degree / mPDR reachability experiment (paper Fig. 9 style):
# sweeps 4 network densities x 4 gravity directions and renders a grouped
# bar plot of average reachability.
python3 gravitational_algorithms.py --nodes 200 --scale fast --experiment degree
python3 gravitational_algorithms.py --nodes 200 --scale full --experiment degree

# Parallel reachability sweep (long runs):
python3 gravitational_algorithms.py --nodes 200 --scale full --workers 12

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

## Parameter choices vs. the paper

- Gravity magnitude: the paper's experiments run under a uniform field of
  magnitude `g` (Earth-like, ~10 m/s²); the code uses `GRAVITY_ACCEL = 10.0`.
- Simulation step `Δt = 0.01 s`, 8 gravity directions at 45° spacing, and the
  boundary-stopping criterion (a hop near the down-field edge that would turn
  back more than 85° from the field is infeasible) follow the paper's
  Algorithms I/II and Sec. 10.2.
- Full-scale sweeps use 200 nodes on a 10×10 plane with a 4×4 regional heat
  map (2.5×2.5 unit regions), matching the paper's experimental setup.

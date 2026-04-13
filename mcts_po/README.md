# MCTS with Partial Observability for Pommerman

This directory extends the existing `mcts/` baseline with partial observability
handling, based on the Locally Interdependent Multi-Agent MDP framework.

## Context

The existing `mcts/` baseline has **full access to game state** — it knows
everything, including what's behind fog. As the original README notes:

> "Handling the partial observability in Pommerman with MCTS is open question."

This module answers that question with three progressively better strategies:

| Agent | PO Strategy | File |
|-------|-------------|------|
| `NaiveMCTSAgent` | Ignore fog — treat hidden cells as empty | `mcts_po_agent.py` |
| `EstimationMCTSAgent` | Track last-known positions, estimate drift | `mcts_po_agent.py` |
| `ParticleMCTSAgent` | Particle filter belief + IS-MCTS determinization | `mcts_po_agent.py` |

## How it works

The existing baseline does:
1. Save game state via `env.get_json_info()`
2. Run MCTS iterations by restoring state and stepping the env
3. Pick the best action from visit counts

Our extension modifies step 1: before MCTS starts, we construct a
**completed state** where fog-of-war cells are filled in with our best
estimate of what's there. The three agents differ in how they fill in fog:

- **Naive**: All fog → passage. Fast but gets surprised by hidden enemies.
- **Estimation**: Remember terrain + track where enemies were last seen.
  Estimate their current position via BFS reachability + drift heuristic.
- **Particle**: Maintain weighted particles per hidden enemy. Sample from
  the particle distribution to create multiple "possible worlds" (determinizations).
  Run MCTS on each and aggregate action scores (Information Set MCTS).

## Connection to theory (Locally Interdependent MDP)

Pommerman maps to the framework as:
- **Metric space**: 11×11 grid, Manhattan distance
- **Speed limit**: 1 cell/step
- **R (dependence radius)**: bomb blast range (~2-6 cells)
- **V (visibility)**: observation window radius (2 cells → 5×5 view)
- **c = ⌊(V-R)/2⌋**: buffer time before hidden agents can affect you

Lemma 4.3 from the mid-report proves that local Nash equilibria within
visibility groups are O(γ^(c+1)) close to the global Nash equilibrium.
Better estimation of out-of-view agents effectively increases the
"planning visibility," tightening this bound in practice.

## Usage

```bash
# Prerequisites: install playground first
cd ../playground && pip install -e . && cd ../pommerman-baselines

# Run evaluation against 3 SimpleAgents (like the existing mcts/ baseline)
python mcts_po/run_mcts_po.py --po_strategy naive --mcts_iters 100 --num_episodes 40 --num_runners 4

python mcts_po/run_mcts_po.py --po_strategy estimation --mcts_iters 100 --num_episodes 40 --num_runners 4

python mcts_po/run_mcts_po.py --po_strategy particle --mcts_iters 100 --num_episodes 40 --num_runners 4

# Quick test with rendering
python mcts_po/run_mcts_po.py --po_strategy estimation --mcts_iters 50 --num_episodes 1 --num_runners 1 --render

# Team mode (partially observable — the primary setting for our project)
python mcts_po/run_mcts_po.py --po_strategy particle --mcts_iters 100 --num_episodes 40 --num_runners 4 --env PommeTeamCompetition-v0
```

## Files

```
mcts_po/
├── README.md                # This file
├── mcts_po_agent.py         # All three MCTS-PO agent classes
├── agent_tracker.py         # Partial observability: tracking & estimation
├── run_mcts_po.py           # Evaluation runner (parallel, like mcts/mcts_agent.py)
└── compare_strategies.py    # Run all 3 strategies and produce comparison table
```

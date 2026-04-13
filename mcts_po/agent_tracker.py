"""
Partial observability handling for Pommerman MCTS.

Three strategies for estimating what's behind fog-of-war:

1. NaiveTracker:     Fog -> passage. Ignore hidden agents.
2. LastKnownTracker: Belief-based tracker using argmax (MAP) estimation.
3. ParticleTracker:  Belief-based tracker using stochastic sampling for IS-MCTS.

All trackers expose the same interface:
    tracker.reset()                  - reset between episodes
    tracker.update(obs)              - feed new observation each step
    tracker.get_estimates()          - {eid: (r,c)} MAP estimate for hidden enemies
    tracker.complete_state(state)    - return a full game state with fog filled in (argmax)
    tracker.sample_state(state)      - sample one possible world from belief
"""

import copy
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from pommerman import constants

BOARD_SIZE = constants.BOARD_SIZE  # 11

# Item values from pommerman.constants.Item
PASSAGE    = constants.Item.Passage.value     # 0
RIGID      = constants.Item.Rigid.value       # 1
WOOD       = constants.Item.Wood.value        # 2
BOMB       = constants.Item.Bomb.value        # 3
FLAMES     = constants.Item.Flames.value      # 4
FOG        = constants.Item.Fog.value         # 5
EXTRA_BOMB = constants.Item.ExtraBomb.value   # 6
INCR_RANGE = constants.Item.IncrRange.value   # 7
KICK       = constants.Item.Kick.value        # 8
AGENT0     = constants.Item.Agent0.value      # 10
AGENT1     = constants.Item.Agent1.value      # 11
AGENT2     = constants.Item.Agent2.value      # 12
AGENT3     = constants.Item.Agent3.value      # 13

AGENT_IDS     = {AGENT0, AGENT1, AGENT2, AGENT3}
DYNAMIC_ITEMS = AGENT_IDS | {BOMB}

# Confidence threshold: only place enemy on MAP estimate if belief mass at peak > this
_MAP_CONFIDENCE_THRESHOLD = 0.05
# Hypothetical bomb weight
_HYPO_BOMB_WEIGHT = 0.3
# Minimum steps hidden before we generate a hypothetical bomb
_HYPO_MIN_STEPS_HIDDEN = 2
# Maximum BFS radius for belief-collapse recovery
_COLLAPSE_BFS_MAX = 20


def _enemy_id_value(enemy):
    """Safely extract the integer value from an enemy reference.

    In pommerman, obs['enemies'] can contain Item enums, raw ints,
    or AgentDummy objects depending on context. Handle all cases.
    """
    if hasattr(enemy, 'value'):
        return enemy.value
    if isinstance(enemy, int):
        return enemy
    return None


def _board_as_int(board_data):
    """Convert board data to a numpy int array.

    get_json_info() returns board as a STRING of a nested list: '[[0, 2, ...], ...]'
    obs['board'] returns a numpy uint8 array.
    Handle both.
    """
    if isinstance(board_data, str):
        board_data = json.loads(board_data)
    return np.array(board_data, dtype=int)


def _board_to_json(board_np):
    """Convert numpy board back to the string format get_json_info() uses."""
    return json.dumps(board_np.tolist())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EnemyTrack:
    """Per-enemy tracking state including a full probability belief grid."""
    enemy_id: int
    alive: bool = True
    last_seen_pos: Optional[Tuple[int, int]] = None
    last_seen_step: Optional[int] = None
    last_seen_dir: Optional[int] = None   # 0-4 (Stop/Up/Down/Left/Right)
    ammo: Optional[int] = None
    blast_strength: Optional[int] = None
    can_kick: Optional[bool] = None
    # Probability mass over (BOARD_SIZE x BOARD_SIZE) cells
    belief: np.ndarray = field(
        default_factory=lambda: np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=float)
    )


@dataclass
class BombTrack:
    """Tracking record for a bomb (confirmed or hypothetical)."""
    pos: Tuple[int, int]
    life: int                       # remaining life, decremented each step
    blast_strength: int
    owner_id: Optional[int] = None
    confirmed: bool = True          # True = directly observed, False = hypothetical
    hypothesis_weight: float = 1.0  # 1.0 for confirmed, 0.0-1.0 for hypothetical
    placed_step: Optional[int] = None


# ---------------------------------------------------------------------------
# BeliefTracker base class
# ---------------------------------------------------------------------------

class BeliefTracker:
    """
    Base class implementing principled hidden-state estimation via a per-enemy
    belief grid maintained through Bayesian filtering.

    Not exposed directly — use LastKnownTracker or ParticleTracker.
    """

    def __init__(self, agent_id: int,
                 use_hypothetical_bombs: bool = False,
                 use_direction_bias: bool = True):
        self.agent_id = agent_id
        self.use_hypothetical_bombs = use_hypothetical_bombs
        self.use_direction_bias = use_direction_bias
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all state between episodes."""
        self.step_count: int = 0
        self.enemy_tracks: Dict[int, EnemyTrack] = {}
        self.bomb_tracks: List[BombTrack] = []
        self.known_board: np.ndarray = np.full(
            (BOARD_SIZE, BOARD_SIZE), FOG, dtype=int
        )
        self._belief_collapses: int = 0
        self.my_pos: Optional[Tuple[int, int]] = None

    def update(self, obs: dict):
        """Feed a new observation dict. Call every step."""
        self.step_count += 1

        # Track our own position for chase-bias in belief propagation
        pos_raw = obs.get('position')
        if pos_raw is not None:
            self.my_pos = (int(pos_raw[0]), int(pos_raw[1]))

        board = _board_as_int(obs['board'])
        bomb_life_obs = np.array(
            obs.get('bomb_life', np.zeros((BOARD_SIZE, BOARD_SIZE))), dtype=int
        )
        bomb_blast_obs = np.array(
            obs.get('bomb_blast_strength', np.zeros((BOARD_SIZE, BOARD_SIZE))), dtype=int
        )

        # 1. Update terrain memory (non-fog, non-dynamic)
        visible_mask = (board != FOG)
        self.known_board[visible_mask] = board[visible_mask]
        # Overwrite dynamic cells in known_board with PASSAGE so terrain is clean
        for eid in AGENT_IDS:
            self.known_board[self.known_board == eid] = PASSAGE
        self.known_board[self.known_board == BOMB] = PASSAGE

        # Build visible-cell set for pruning
        visible_cells: set = set()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if visible_mask[r, c]:
                    visible_cells.add((r, c))

        # 2. Parse the enemy list from observation
        enemies = obs.get('enemies', [])
        enemy_ids_tracked: set = set()
        for enemy in enemies:
            eid = _enemy_id_value(enemy)
            if eid is None:
                continue
            enemy_ids_tracked.add(eid)

            if eid not in self.enemy_tracks:
                self.enemy_tracks[eid] = EnemyTrack(enemy_id=eid)

            track = self.enemy_tracks[eid]
            positions = np.argwhere(board == eid)

            if len(positions) > 0:
                # Enemy is VISIBLE this step
                pos = (int(positions[0][0]), int(positions[0][1]))

                # Check for direction if we have a previous position
                if track.last_seen_pos is not None:
                    dr = pos[0] - track.last_seen_pos[0]
                    dc = pos[1] - track.last_seen_pos[1]
                    if dr == 0 and dc == 0:
                        track.last_seen_dir = 0   # Stop
                    elif dr == -1:
                        track.last_seen_dir = 1   # Up
                    elif dr == 1:
                        track.last_seen_dir = 2   # Down
                    elif dc == -1:
                        track.last_seen_dir = 3   # Left
                    elif dc == 1:
                        track.last_seen_dir = 4   # Right

                track.last_seen_pos = pos
                track.last_seen_step = self.step_count
                track.alive = True

                # Refresh attributes if available in obs
                if 'teammate' in obs and hasattr(obs.get('teammate'), 'value'):
                    pass  # no per-enemy ammo in obs dict easily; skip for now
                # Pull ammo/blast/kick from agent's own obs if it matches
                # (only reliable for self; for others these are not in standard obs)

                # Reset belief to a delta at observed position
                track.belief[:] = 0.0
                track.belief[pos[0], pos[1]] = 1.0

            else:
                # Enemy is NOT visible this step
                if not track.alive:
                    continue

                # Check if the enemy was last seen in a now-visible cell with flames
                if track.last_seen_pos is not None:
                    lr, lc = track.last_seen_pos
                    if (lr, lc) in visible_cells:
                        cell_val = int(board[lr, lc])
                        if cell_val == FLAMES:
                            track.alive = False
                            track.belief[:] = 0.0
                            continue

                # Propagate belief
                self._propagate_belief(track)

                # Prune by negative evidence (visibility)
                self._prune_belief(track, visible_cells, board)

                # Prune by danger zones (confirmed bomb blast radii)
                self._danger_zone_prune_belief(track, board)

                # Renormalize; recover from collapse
                total = track.belief.sum()
                if total < 1e-9:
                    self._belief_collapses += 1
                    self._reinitialize_from_last_seen(track)
                else:
                    track.belief /= total

        # 3. Update bomb tracks
        self._update_bombs(board, bomb_life_obs, bomb_blast_obs, visible_cells)

        # 4. Hypothetical bombs (optional)
        if self.use_hypothetical_bombs:
            self._generate_hypothetical_bombs()

    def get_estimates(self) -> Dict[int, Tuple[int, int]]:
        """Return {enemy_tile_value: (row, col)} MAP estimate for each hidden enemy."""
        estimates = {}
        for eid, track in self.enemy_tracks.items():
            if not track.alive:
                continue
            if track.last_seen_step == self.step_count:
                # Visible this step — not hidden
                continue
            if track.belief.sum() < 1e-9:
                continue
            r, c = np.unravel_index(np.argmax(track.belief), track.belief.shape)
            estimates[eid] = (int(r), int(c))
        return estimates

    def get_debug_info(self) -> dict:
        """Return debug information for optional logging."""
        hidden_enemies = {}
        for eid, track in self.enemy_tracks.items():
            if not track.alive:
                continue
            if track.last_seen_step == self.step_count:
                continue
            total = track.belief.sum()
            if total < 1e-9:
                continue
            prob = track.belief / total
            entropy = float(-np.sum(prob * np.log(prob + 1e-12)))
            support = int(np.sum(prob > 1e-6))
            r, c = np.unravel_index(np.argmax(prob), prob.shape)
            # Top-3 positions
            flat_idx = np.argsort(prob.ravel())[::-1][:3]
            top3 = [
                (int(i // BOARD_SIZE), int(i % BOARD_SIZE), float(prob.ravel()[i]))
                for i in flat_idx
            ]
            hidden_enemies[eid] = {
                'map_pos': (int(r), int(c)),
                'entropy': entropy,
                'support_size': support,
                'top3': top3,
            }

        confirmed_hidden = sum(
            1 for b in self.bomb_tracks if b.confirmed and b.life > 0
        )
        hypothetical = sum(
            1 for b in self.bomb_tracks if not b.confirmed
        )

        return {
            'hidden_enemies': hidden_enemies,
            'confirmed_hidden_bombs': confirmed_hidden,
            'hypothetical_bombs': hypothetical,
            'belief_collapses': self._belief_collapses,
        }

    def complete_state(self, fogged_state: dict) -> dict:
        """
        Fill fog using terrain memory and place enemies at MAP (argmax) positions.
        Returns a state dict with 'board' as a JSON string.
        """
        state = copy.deepcopy(fogged_state)
        board = _board_as_int(state['board'])

        # Fill fog with known terrain
        board = self._fill_fog_terrain(board)

        # Find which agents are already visible on the board
        visible_agents = self._get_visible_agents(board)

        # Place each hidden enemy at the MAP position
        for eid, track in self.enemy_tracks.items():
            if not track.alive:
                continue
            if eid in visible_agents:
                continue
            if track.belief.sum() < 1e-9:
                continue
            peak_mass = track.belief.max()
            if peak_mass < _MAP_CONFIDENCE_THRESHOLD:
                continue
            r, c = np.unravel_index(np.argmax(track.belief), track.belief.shape)
            if int(board[r, c]) == PASSAGE:
                board[r, c] = eid

        # Insert confirmed hidden bombs
        self._insert_confirmed_bombs(board)

        # Insert hypothetical bombs above weight threshold (if enabled)
        if self.use_hypothetical_bombs:
            self._insert_hypothetical_bombs(board, threshold=0.5)

        state['board'] = _board_to_json(board)
        return state

    def sample_state(self, fogged_state: dict) -> dict:
        """
        Fill fog using terrain memory and sample enemy positions from belief.
        Returns a state dict with 'board' as a JSON string.
        """
        state = copy.deepcopy(fogged_state)
        board = _board_as_int(state['board'])

        # Fill fog with known terrain
        board = self._fill_fog_terrain(board)

        # Find which agents are already visible on the board
        visible_agents = self._get_visible_agents(board)

        # Sample positions for each hidden enemy; resolve conflicts afterwards
        sampled_positions: Dict[int, Tuple[int, int]] = {}
        belief_masses: Dict[int, float] = {}

        for eid, track in self.enemy_tracks.items():
            if not track.alive:
                continue
            if eid in visible_agents:
                continue
            total = track.belief.sum()
            if total < 1e-9:
                continue
            prob = track.belief / total
            flat_idx = np.random.choice(BOARD_SIZE * BOARD_SIZE, p=prob.ravel())
            r = flat_idx // BOARD_SIZE
            c = flat_idx % BOARD_SIZE
            sampled_positions[eid] = (int(r), int(c))
            belief_masses[eid] = float(prob.ravel()[flat_idx])

        # Resolve conflicts: two enemies at same cell
        occupied: Dict[Tuple[int, int], int] = {}
        placed_order = sorted(sampled_positions.keys(),
                              key=lambda e: -belief_masses.get(e, 0.0))
        for eid in placed_order:
            pos = sampled_positions[eid]
            if pos in occupied:
                # Resample from belief excluding already-occupied cells
                track = self.enemy_tracks[eid]
                total = track.belief.sum()
                if total < 1e-9:
                    continue
                prob = track.belief.copy() / total
                # Zero out occupied positions and renormalize
                for opos in occupied:
                    prob[opos[0], opos[1]] = 0.0
                if prob.sum() < 1e-9:
                    continue
                prob /= prob.sum()
                flat_idx = np.random.choice(BOARD_SIZE * BOARD_SIZE, p=prob.ravel())
                pos = (flat_idx // BOARD_SIZE, flat_idx % BOARD_SIZE)
                sampled_positions[eid] = pos

            if int(board[pos[0], pos[1]]) == PASSAGE:
                board[pos[0], pos[1]] = eid
                occupied[pos] = eid

        # Insert confirmed hidden bombs
        self._insert_confirmed_bombs(board)

        # Sample hypothetical bomb placements if enabled
        if self.use_hypothetical_bombs:
            self._sample_hypothetical_bombs(board)

        state['board'] = _board_to_json(board)
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_legal_cell(self, r: int, c: int) -> bool:
        """True if (r,c) is in bounds and not rigid/wood in known_board."""
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return False
        cell = self.known_board[r, c]
        return cell not in (RIGID, WOOD)

    def _legal_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Return list of cells reachable in one step from (r,c) on known_board."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self._is_legal_cell(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def _propagate_belief(self, track: EnemyTrack):
        """
        Propagate one time step using a behaviorally-informed motion prior.

        Instead of uniform weights, each candidate cell is weighted by:
        - Direction bias: continuing last known direction is more likely
        - Wood proximity: enemies tend to move toward wood (to bomb it)
        - Chase bias: enemies tend to close distance toward our agent
        Stop action is downweighted when the enemy was last seen moving.
        """
        old_belief = track.belief.copy()
        new_belief = np.zeros_like(old_belief)

        # Stop is less likely if enemy was seen moving
        base_stop_w = 0.6 if (self.use_direction_bias
                               and track.last_seen_dir is not None
                               and track.last_seen_dir != 0) else 1.0

        # Direction encoding: (dr, dc) for dirs 1-4 (Up/Down/Left/Right)
        _DIR_VEC = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

        nonzero_cells = np.argwhere(old_belief > 0)
        for idx in nonzero_cells:
            r, c = int(idx[0]), int(idx[1])
            mass = old_belief[r, c]
            moves = self._legal_neighbors(r, c)

            # --- Compute per-destination weights ---
            dest_weights: Dict[Tuple[int, int], float] = {}

            # Stop
            dest_weights[(r, c)] = base_stop_w

            for nr, nc in moves:
                w = 1.0

                # 1. Direction bias: boost the cell that continues last known direction
                if (self.use_direction_bias
                        and track.last_seen_dir is not None
                        and track.last_seen_dir in _DIR_VEC):
                    exp_dr, exp_dc = _DIR_VEC[track.last_seen_dir]
                    if (nr - r) == exp_dr and (nc - c) == exp_dc:
                        w += 0.5

                # 2. Wood-proximity bias: enemy tends to move toward wood
                for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    wr, wc = nr + ddr, nc + ddc
                    if (0 <= wr < BOARD_SIZE and 0 <= wc < BOARD_SIZE
                            and self.known_board[wr, wc] == WOOD):
                        w += 0.3
                        break  # one adjacent wood is enough

                # 3. Chase bias: enemy tends to close distance to our agent
                if self.my_pos is not None:
                    my_r, my_c = self.my_pos
                    curr_dist = abs(r - my_r) + abs(c - my_c)
                    new_dist  = abs(nr - my_r) + abs(nc - my_c)
                    if new_dist < curr_dist:
                        w += 0.4

                dest_weights[(nr, nc)] = w

            total_w = sum(dest_weights.values())
            if total_w <= 0:
                new_belief[r, c] += mass
                continue

            for (dr, dc), w in dest_weights.items():
                new_belief[dr, dc] += mass * (w / total_w)

        track.belief = new_belief

    def _prune_belief(self, track: EnemyTrack, visible_cells: set, board: np.ndarray):
        """
        Zero out belief mass on cells that are currently visible and do NOT
        contain the target enemy (negative evidence).
        """
        eid = track.enemy_id
        for r, c in visible_cells:
            if int(board[r, c]) != eid:
                track.belief[r, c] = 0.0

    def _blast_zone(self, bomb_r: int, bomb_c: int,
                    blast_strength: int, board: np.ndarray) -> set:
        """Return the set of cells in the blast radius of a bomb at (bomb_r, bomb_c).

        Uses known_board terrain to block blasts at RIGID walls and stop at WOOD
        (matching Pommerman blast propagation rules).
        """
        zone = {(bomb_r, bomb_c)}
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, blast_strength + 1):
                nr, nc = bomb_r + dr * dist, bomb_c + dc * dist
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    break
                cell = self.known_board[nr, nc]
                if cell == RIGID:
                    break        # rigid blocks blast entirely
                zone.add((nr, nc))
                if cell == WOOD:
                    break        # wood absorbs blast (doesn't propagate past)
        return zone

    def _danger_zone_prune_belief(self, track: EnemyTrack, board: np.ndarray):
        """Suppress belief mass in the blast radius of confirmed active bombs.

        Enemies are unlikely to voluntarily stand in an active blast zone, so we
        heavily discount (not zero out) those cells. Cells are not zeroed because
        the enemy might be fleeing through them.
        """
        for bt in self.bomb_tracks:
            if not bt.confirmed:
                continue
            zone = self._blast_zone(bt.pos[0], bt.pos[1], bt.blast_strength, board)
            for r, c in zone:
                track.belief[r, c] *= 0.1

    def _has_escape_route(self, pos: Tuple[int, int], blast_strength: int) -> bool:
        """Check whether placing a bomb at pos leaves at least one escape route.

        Mirrors SimpleAgent's _maybe_bomb logic: an escape exists if the enemy
        can reach a passage cell that is outside the bomb's blast zone.
        """
        blast_zone = self._blast_zone(pos[0], pos[1], blast_strength, self.known_board)

        # BFS from pos on known_board
        visited = {pos}
        frontier = deque([pos])

        while frontier:
            cr, cc = frontier.popleft()
            for nr, nc in self._legal_neighbors(cr, cc):
                if (nr, nc) in visited:
                    continue
                visited.add((nr, nc))
                if (nr, nc) not in blast_zone:
                    return True   # reachable cell outside blast range → safe
                frontier.append((nr, nc))

        return False

    def _reinitialize_from_last_seen(self, track: EnemyTrack):
        """
        Belief collapsed to zero. Reinitialize from last_seen_pos via BFS
        on known_board, up to min(steps_since_seen, _COLLAPSE_BFS_MAX) steps.
        """
        if track.last_seen_pos is None:
            # Never seen: uniform over all legal cells
            legal = []
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if self._is_legal_cell(r, c):
                        legal.append((r, c))
            track.belief[:] = 0.0
            if legal:
                for r, c in legal:
                    track.belief[r, c] = 1.0 / len(legal)
            return

        steps_since = (
            self.step_count - track.last_seen_step
            if track.last_seen_step is not None
            else _COLLAPSE_BFS_MAX
        )
        max_dist = min(steps_since, _COLLAPSE_BFS_MAX)

        reachable = self._bfs_reachable(
            track.last_seen_pos[0], track.last_seen_pos[1], max_dist
        )

        track.belief[:] = 0.0
        if reachable:
            for r, c in reachable:
                track.belief[r, c] = 1.0 / len(reachable)
        else:
            # Fallback: place mass at last seen pos even if it looks blocked
            lr, lc = track.last_seen_pos
            track.belief[lr, lc] = 1.0

    def _bfs_reachable(self, start_r: int, start_c: int,
                       max_dist: int) -> List[Tuple[int, int]]:
        """BFS on known_board; returns all cells reachable within max_dist steps."""
        visited = {(start_r, start_c)}
        frontier = deque([(start_r, start_c)])
        depth = {(start_r, start_c): 0}

        while frontier:
            r, c = frontier.popleft()
            if depth[(r, c)] >= max_dist:
                continue
            for nr, nc in self._legal_neighbors(r, c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    depth[(nr, nc)] = depth[(r, c)] + 1
                    frontier.append((nr, nc))

        return list(visited)

    def _update_bombs(self, board: np.ndarray, bomb_life_obs: np.ndarray,
                      bomb_blast_obs: np.ndarray, visible_cells: set):
        """
        Refresh bomb_tracks from current observation:
        - add/update confirmed bombs that are currently visible
        - decrement life on confirmed hidden bombs
        - prune expired bombs
        """
        # Collect currently visible bomb positions from the board
        visible_bomb_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if int(board[r, c]) == BOMB and (r, c) in visible_cells:
                    life_val = int(bomb_life_obs[r, c]) if bomb_life_obs[r, c] > 0 else 9
                    blast_val = int(bomb_blast_obs[r, c]) if bomb_blast_obs[r, c] > 0 else 2
                    visible_bomb_positions[(r, c)] = (life_val, blast_val)

        # Mark existing confirmed tracks as seen or decrement
        confirmed_positions = {b.pos for b in self.bomb_tracks if b.confirmed}
        new_bomb_tracks: List[BombTrack] = []

        for bt in self.bomb_tracks:
            if bt.confirmed:
                if bt.pos in visible_bomb_positions:
                    # Refresh life from observation
                    life_val, blast_val = visible_bomb_positions[bt.pos]
                    bt.life = life_val
                    bt.blast_strength = blast_val
                    new_bomb_tracks.append(bt)
                    visible_bomb_positions.pop(bt.pos)
                else:
                    # Hidden confirmed bomb: decrement life
                    if bt.pos in visible_cells:
                        # Was visible but bomb is gone — it exploded or was invalid
                        pass  # drop it
                    else:
                        bt.life -= 1
                        if bt.life > 0:
                            new_bomb_tracks.append(bt)
            else:
                # Hypothetical bomb: keep if weight is meaningful and not expired
                bt.life -= 1
                if bt.life > 0 and bt.hypothesis_weight > 0.05:
                    # If the hypothetical bomb's location became visible, remove it
                    if bt.pos not in visible_cells:
                        new_bomb_tracks.append(bt)

        # Add newly observed bombs not in existing tracks
        for (r, c), (life_val, blast_val) in visible_bomb_positions.items():
            new_bomb_tracks.append(BombTrack(
                pos=(r, c),
                life=life_val,
                blast_strength=blast_val,
                owner_id=None,
                confirmed=True,
                hypothesis_weight=1.0,
                placed_step=self.step_count,
            ))

        self.bomb_tracks = new_bomb_tracks

    def _generate_hypothetical_bombs(self):
        """
        For each hidden enemy with ammo > 0 that has been out of sight for
        at least _HYPO_MIN_STEPS_HIDDEN steps and was last seen near wood,
        add a low-weight hypothetical BombTrack if none exists yet.
        """
        existing_hypo_owners = {b.owner_id for b in self.bomb_tracks if not b.confirmed}

        for eid, track in self.enemy_tracks.items():
            if not track.alive:
                continue
            if track.last_seen_step == self.step_count:
                continue  # visible this step
            if track.ammo is not None and track.ammo <= 0:
                continue  # no ammo
            if eid in existing_hypo_owners:
                continue  # already have one

            if track.last_seen_pos is None:
                continue

            steps_hidden = self.step_count - (track.last_seen_step or 0)
            if steps_hidden < _HYPO_MIN_STEPS_HIDDEN:
                continue

            # Check if last seen near wood within blast range (default 2)
            blast = track.blast_strength if track.blast_strength is not None else 2
            lr, lc = track.last_seen_pos
            near_wood = False
            for dr in range(-blast, blast + 1):
                for dc in range(-blast, blast + 1):
                    nr, nc = lr + dr, lc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if self.known_board[nr, nc] == WOOD:
                            near_wood = True
                            break
                if near_wood:
                    break

            if not near_wood:
                continue

            # Use MAP position of enemy belief as hypothetical bomb position
            if track.belief.sum() < 1e-9:
                continue
            r, c = np.unravel_index(np.argmax(track.belief), track.belief.shape)

            # Only generate hypothesis if placing a bomb here would leave an escape
            # route (mirrors SimpleAgent's _maybe_bomb check — enemies don't
            # self-trap, so if there's no escape they wouldn't plant)
            if not self._has_escape_route((int(r), int(c)), blast):
                continue

            self.bomb_tracks.append(BombTrack(
                pos=(int(r), int(c)),
                life=9,
                blast_strength=blast,
                owner_id=eid,
                confirmed=False,
                hypothesis_weight=_HYPO_BOMB_WEIGHT,
                placed_step=self.step_count,
            ))

    def _fill_fog_terrain(self, board: np.ndarray) -> np.ndarray:
        """Fill fogged cells with known terrain (non-dynamic only)."""
        fog_mask = (board == FOG)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if fog_mask[r, c]:
                    mem = self.known_board[r, c]
                    if mem != FOG and mem not in DYNAMIC_ITEMS:
                        board[r, c] = mem
                    else:
                        board[r, c] = PASSAGE
        return board

    def _get_visible_agents(self, board: np.ndarray) -> set:
        """Return set of agent tile values visible on the (already fog-filled) board."""
        visible = set()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = int(board[r, c])
                if val in AGENT_IDS:
                    visible.add(val)
        return visible

    def _insert_confirmed_bombs(self, board: np.ndarray):
        """Place confirmed hidden bombs (life > 0) back onto the board."""
        for bt in self.bomb_tracks:
            if not bt.confirmed:
                continue
            r, c = bt.pos
            # Only place if that cell is currently passage (fog was cleared above)
            if int(board[r, c]) == PASSAGE:
                board[r, c] = BOMB

    def _insert_hypothetical_bombs(self, board: np.ndarray, threshold: float = 0.5):
        """Place hypothetical bombs above weight threshold."""
        for bt in self.bomb_tracks:
            if bt.confirmed:
                continue
            if bt.hypothesis_weight < threshold:
                continue
            r, c = bt.pos
            if int(board[r, c]) == PASSAGE:
                board[r, c] = BOMB

    def _sample_hypothetical_bombs(self, board: np.ndarray):
        """Sample hypothetical bomb placements based on hypothesis_weight."""
        for bt in self.bomb_tracks:
            if bt.confirmed:
                continue
            if np.random.random() < bt.hypothesis_weight:
                r, c = bt.pos
                if int(board[r, c]) == PASSAGE:
                    board[r, c] = BOMB


# ---------------------------------------------------------------------------
# Public tracker classes
# ---------------------------------------------------------------------------

class NaiveTracker:
    """Phase 1: Replace all fog with passage. Ignore hidden agents entirely."""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def reset(self):
        pass

    def update(self, obs: dict):
        pass

    def get_estimates(self) -> dict:
        """Return estimated positions for hidden enemies. Empty for naive strategy."""
        return {}

    def complete_state(self, json_state: dict) -> dict:
        """Return a modified json_state with fog replaced by passage."""
        state = copy.deepcopy(json_state)
        board = _board_as_int(state['board'])
        board[board == FOG] = PASSAGE
        state['board'] = _board_to_json(board)
        return state

    def sample_state(self, json_state: dict) -> dict:
        return self.complete_state(json_state)


class LastKnownTracker(BeliefTracker):
    """
    Belief-based tracker using MAP (argmax) estimation for state completion.

    Uses BeliefTracker's belief grid; complete_state places enemies at the
    peak of each belief distribution. sample_state falls back to complete_state
    (deterministic argmax).
    """

    def __init__(self, agent_id: int,
                 use_hypothetical_bombs: bool = False,
                 use_direction_bias: bool = True):
        super().__init__(
            agent_id=agent_id,
            use_hypothetical_bombs=use_hypothetical_bombs,
            use_direction_bias=use_direction_bias,
        )

    def sample_state(self, fogged_state: dict) -> dict:
        """LastKnownTracker uses argmax for both complete and sample."""
        return self.complete_state(fogged_state)


class ParticleTracker(BeliefTracker):
    """
    Belief-based tracker using stochastic sampling for IS-MCTS.

    Uses BeliefTracker's belief grid; sample_state samples enemy positions
    from the belief distribution, giving different possible worlds per
    determinization. Hypothetical bombs enabled by default.
    """

    def __init__(self, agent_id: int,
                 num_particles: int = 50,
                 use_hypothetical_bombs: bool = True,
                 use_direction_bias: bool = True):
        # num_particles kept for API compatibility; belief grid is used instead
        self.num_particles = num_particles
        super().__init__(
            agent_id=agent_id,
            use_hypothetical_bombs=use_hypothetical_bombs,
            use_direction_bias=use_direction_bias,
        )

"""
MCTS agents with partial observability handling for Pommerman.

This follows the same architecture as the existing mcts/mcts_agent.py
in pommerman-baselines:
  - Uses the real pommerman environment as the forward model
  - Saves/restores state via env.get_json_info() / env._init_game_state
  - AlphaGoZero-style MCTS with UCB exploration

Key addition: before running MCTS iterations, the game state is "completed"
by filling in fog-of-war cells using one of three strategies (naive,
estimation, particle filter). For the particle strategy, we additionally
run IS-MCTS over multiple sampled determinizations.
"""

import copy
import json
import numpy as np
import pommerman
from pommerman import constants
from pommerman.agents import BaseAgent

from agent_tracker import NaiveTracker, LastKnownTracker, ParticleTracker

NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action)


def argmax_tiebreaking(Q):
    """Find the best action with random tie-breaking."""
    idx = np.flatnonzero(Q == np.max(Q))
    assert len(idx) > 0, str(Q)
    return np.random.choice(idx)


class MCTSNode:
    """
    MCTS tree node. Stores per-action statistics for our agent.
    """

    def __init__(self):
        self.Q = np.zeros(NUM_ACTIONS)
        self.W = np.zeros(NUM_ACTIONS)
        self.N = np.zeros(NUM_ACTIONS, dtype=np.uint32)


class MCTSPOAgent:
    """
    MCTS agent with partial observability handling.

    Args:
        agent_id: Which agent we control (0-3)
        po_strategy: 'naive', 'estimation', or 'particle'
        num_determinizations: How many worlds to sample (particle only)
        c_puct: Exploration constant for UCB
    """

    def __init__(self, agent_id=0, po_strategy='naive',
                 num_determinizations=5, c_puct=1.0,
                 env_id='PommeFFACompetition-v0'):
        self.agent_id = agent_id
        self.po_strategy = po_strategy
        self.num_determinizations = num_determinizations
        self.c_puct = c_puct
        self.env_id = env_id

        # Create a private env instance for forward simulation
        self.env = self._make_env()
        self.tree = {}
        self.last_obs = None

        # Create the appropriate tracker
        if po_strategy == 'naive':
            self.tracker = NaiveTracker(agent_id)
        elif po_strategy == 'estimation':
            self.tracker = LastKnownTracker(agent_id)
        elif po_strategy == 'particle':
            self.tracker = ParticleTracker(agent_id, num_particles=50)
        else:
            raise ValueError(f"Unknown PO strategy: {po_strategy}")

    def _make_env(self):
        """Create a private env for MCTS simulation."""
        agents = [BaseAgent() for _ in range(NUM_AGENTS)]
        env = pommerman.make(self.env_id, agents)
        return env

    def reset(self):
        """Reset between episodes."""
        self.tree = {}
        self.last_obs = None
        self.tracker.reset()

    def update_tracker(self, obs):
        """Feed a new observation to the tracker. Call this every step."""
        self.last_obs = obs
        self.tracker.update(obs)

    def get_tracker_estimates(self):
        """Return {eid: estimated_pos} from the tracker for logging."""
        return self.tracker.get_estimates()

    def _apply_observation_fog(self, root_state):
        """Mask root_state with fog from our actual partial observation.

        get_json_info() returns the true board with no fog. We replace cells
        that are FOG in our observation with FOG in the state, so the tracker
        only fills in cells the agent genuinely cannot see.
        """
        if self.last_obs is None:
            return root_state

        state = copy.deepcopy(root_state)
        board_data = state['board']
        if isinstance(board_data, str):
            board = np.array(json.loads(board_data), dtype=int)
        else:
            board = np.array(board_data, dtype=int)

        obs_board = np.array(self.last_obs['board'], dtype=int)
        fog_val = constants.Item.Fog.value
        board[obs_board == fog_val] = fog_val

        state['board'] = json.dumps(board.tolist())
        return state

    def search(self, root_state, num_iters, temperature=1.0):
        """
        Run MCTS from the given root state and return action + probabilities.

        Args:
            root_state: The real game state from env.get_json_info()
            num_iters: Total MCTS iterations budget
            temperature: 0 = greedy, 1 = proportional to visit counts.

        Returns:
            (action, action_probs)
        """
        fogged_state = self._apply_observation_fog(root_state)
        if self.po_strategy == 'particle':
            return self._search_is_mcts(fogged_state, num_iters, temperature)
        else:
            completed = self.tracker.complete_state(fogged_state)
            return self._search_single(completed, num_iters, temperature)

    def _search_single(self, state, num_iters, temperature):
        """Standard MCTS on a single (completed) state."""
        self.tree = {}

        for _ in range(num_iters):
            # Restore to root
            self.env._init_game_state = copy.deepcopy(state)
            try:
                obs = self.env.reset()
            except Exception:
                continue

            current_key = str(self.env.get_json_info())
            root_key = current_key

            path = []
            done = False
            rewards = [0] * NUM_AGENTS

            for depth in range(40):
                if current_key not in self.tree:
                    self.tree[current_key] = MCTSNode()
                    value = self._evaluate(obs, rewards, done)
                    break

                node = self.tree[current_key]
                action = self._select_action(node)
                path.append((current_key, action))

                actions = self._get_all_actions(action)

                try:
                    obs, rewards, done, info = self.env.step(actions)
                except Exception:
                    value = -0.5
                    break

                if done:
                    value = self._evaluate(obs, rewards, done)
                    break

                current_key = str(self.env.get_json_info())
            else:
                value = self._evaluate(obs, rewards, done)

            # Backpropagation
            for state_key, act in reversed(path):
                node = self.tree[state_key]
                node.N[act] += 1
                node.W[act] += value
                node.Q[act] = node.W[act] / node.N[act]

        # Extract action from root
        root_node = self.tree.get(root_key) if 'root_key' in dir() else None
        # Try to find any root node from the tree
        if root_node is None and self.tree:
            # The first key added is the root
            first_key = next(iter(self.tree))
            root_node = self.tree[first_key]

        if root_node is None:
            return 0, np.ones(NUM_ACTIONS) / NUM_ACTIONS

        return self._pick_action(root_node, temperature)

    def _search_is_mcts(self, root_state, num_iters, temperature):
        """
        Information Set MCTS for particle filter strategy.
        """
        total_N = np.zeros(NUM_ACTIONS)
        total_W = np.zeros(NUM_ACTIONS)
        iters_per_det = max(1, num_iters // self.num_determinizations)

        for _ in range(self.num_determinizations):
            sampled_state = self.tracker.sample_state(root_state)
            self.tree = {}
            root_key = None

            for _ in range(iters_per_det):
                self.env._init_game_state = copy.deepcopy(sampled_state)
                try:
                    obs = self.env.reset()
                except Exception:
                    continue

                current_key = str(self.env.get_json_info())
                if root_key is None:
                    root_key = current_key

                path = []
                done = False
                rewards = [0] * NUM_AGENTS

                for depth in range(40):
                    if current_key not in self.tree:
                        self.tree[current_key] = MCTSNode()
                        value = self._evaluate(obs, rewards, done)
                        break

                    node = self.tree[current_key]
                    action = self._select_action(node)
                    path.append((current_key, action))
                    actions = self._get_all_actions(action)

                    try:
                        obs, rewards, done, info = self.env.step(actions)
                    except Exception:
                        value = -0.5
                        break

                    if done:
                        value = self._evaluate(obs, rewards, done)
                        break
                    current_key = str(self.env.get_json_info())
                else:
                    value = self._evaluate(obs, rewards, done)

                for state_key, act in reversed(path):
                    node = self.tree[state_key]
                    node.N[act] += 1
                    node.W[act] += value
                    node.Q[act] = node.W[act] / node.N[act]

            # Aggregate root visit counts
            if root_key is not None and root_key in self.tree:
                root_node = self.tree[root_key]
                total_N += root_node.N
                total_W += root_node.W

        # Pick action from aggregated statistics
        if total_N.sum() == 0:
            return 0, np.ones(NUM_ACTIONS) / NUM_ACTIONS

        if temperature == 0:
            action = argmax_tiebreaking(total_N)
            probs = np.zeros(NUM_ACTIONS)
            probs[action] = 1.0
        else:
            counts = total_N ** (1.0 / temperature)
            total = counts.sum()
            probs = counts / total if total > 0 else np.ones(NUM_ACTIONS) / NUM_ACTIONS
            action = np.random.choice(NUM_ACTIONS, p=probs)

        return action, probs

    def _select_action(self, node):
        """UCB1 action selection."""
        total_visits = node.N.sum()
        if total_visits == 0:
            return np.random.randint(NUM_ACTIONS)

        ucb = np.full(NUM_ACTIONS, float('inf'))
        visited = node.N > 0
        ucb[visited] = (node.Q[visited] +
                        self.c_puct * np.sqrt(np.log(total_visits) / node.N[visited]))
        return argmax_tiebreaking(ucb)

    def _evaluate(self, obs, rewards, done):
        """Evaluate a leaf node with shaped rewards."""
        if done:
            return rewards[self.agent_id]

        my_obs = obs[self.agent_id] if isinstance(obs, list) else obs
        value = 0.0

        if not isinstance(my_obs, dict):
            return 0.0

        board = np.array(my_obs.get('board', []))
        pos = my_obs.get('position', (0, 0))
        ammo = my_obs.get('ammo', 1)
        blast_strength = my_obs.get('blast_strength', 2)
        bomb_life = np.array(my_obs.get('bomb_life', np.zeros_like(board)))

        if board.size == 0:
            return 0.0

        r, c = pos

        # Survival bonus — being alive is good
        value += 0.1

        # Danger penalty — bombs about to explode nearby
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(0,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if bomb_life[nr, nc] > 0 and bomb_life[nr, nc] <= 3:
                    value -= 0.5
                # Standing in flames is very bad
                if board[nr, nc] == constants.Item.Flames.value and (dr, dc) == (0, 0):
                    value -= 1.0

        # Power-up bonus — stronger agent survives longer
        value += 0.02 * (ammo - 1)
        value += 0.02 * (blast_strength - 2)

        # Mobility — count how many adjacent cells are passable
        passable = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
                if board[nr, nc] in (0, 6, 7, 8):  # passage or power-ups
                    passable += 1
        value += 0.03 * passable  # More escape routes = safer

        return value
    def _get_all_actions(self, my_action):
        """Build action list for all 4 agents.
        
        Opponents use Stop action as a conservative estimate.
        Random actions cause unrealistic simulations where opponents
        suicide-bomb themselves, making MCTS overestimate safety.
        """
        actions = []
        for i in range(NUM_AGENTS):
            if i == self.agent_id:
                actions.append(my_action)
            else:
                # Stop is more realistic than random for short rollouts.
                # Random opponents place bombs everywhere and die instantly,
                # which makes MCTS think the board is safer than it really is.
                actions.append(0)  # Action.Stop
        return actions

    def _pick_action(self, root_node, temperature):
        """Pick final action from root node visit counts."""
        if temperature == 0:
            action = argmax_tiebreaking(root_node.N)
            probs = np.zeros(NUM_ACTIONS)
            probs[action] = 1.0
        else:
            counts = root_node.N.astype(float)
            counts = counts ** (1.0 / temperature)
            total = counts.sum()
            if total > 0:
                probs = counts / total
            else:
                probs = np.ones(NUM_ACTIONS) / NUM_ACTIONS
            action = np.random.choice(NUM_ACTIONS, p=probs)
        return action, probs
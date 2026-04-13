#!/usr/bin/env python3
"""
Evaluation runner for MCTS-PO agents.

Usage:
    python mcts_po/run_mcts_po.py --po_strategy naive --mcts_iters 100 --num_episodes 40 --num_runners 4
    python mcts_po/run_mcts_po.py --po_strategy estimation --mcts_iters 100 --num_episodes 40 --num_runners 4
    python mcts_po/run_mcts_po.py --po_strategy particle --mcts_iters 100 --num_episodes 40 --num_runners 4
    python mcts_po/run_mcts_po.py --po_strategy naive --num_episodes 1 --num_runners 1 --render
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pommerman
from pommerman import constants
from pommerman.agents import SimpleAgent

# Add this directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcts_po_agent import MCTSPOAgent

NUM_AGENTS = 4
NUM_ACTIONS = len(constants.Action)

# Maps agent tile value (10-13) to agent index (0-3)
_AGENT_TILE_TO_ID = {10: 0, 11: 1, 12: 2, 13: 3}


def _get_true_positions(root_state):
    """Parse true agent positions from the full (unfogged) json state.

    Returns {agent_id: (row, col)} for every agent still on the board.
    """
    board_data = root_state['board']
    if isinstance(board_data, str):
        board = np.array(json.loads(board_data), dtype=int)
    else:
        board = np.array(board_data, dtype=int)

    positions = {}
    for tile, aid in _AGENT_TILE_TO_ID.items():
        locs = np.argwhere(board == tile)
        if len(locs) > 0:
            positions[aid] = tuple(int(x) for x in locs[0])
    return positions


def _log_step(step, agent_id, po_strategy, action, all_actions,
               root_state, mcts_agent):
    """Print a one-step summary: positions, actions, and tracker estimates."""
    true_pos = _get_true_positions(root_state)

    my_pos  = true_pos.get(agent_id, '?')
    act_name = constants.Action(action).name
    print(f"  Step {step:>3} | MCTS A{agent_id} ({po_strategy}) @ {my_pos} -> {act_name}")

    # Other agents' true positions and their chosen actions
    others = []
    for i in range(NUM_AGENTS):
        if i == agent_id:
            continue
        pos      = true_pos.get(i, 'dead')
        oact     = constants.Action(all_actions[i]).name
        others.append(f"A{i}@{pos}->{oact}")
    print(f"           Others:  {' | '.join(others)}")

    # Tracker estimates vs truth (only meaningful for estimation / particle)
    if po_strategy in ('estimation', 'particle'):
        estimates = mcts_agent.get_tracker_estimates()
        if not estimates:
            print(f"           Tracker: no hidden enemies tracked yet")
        else:
            parts = []
            for eid, est_pos in sorted(estimates.items()):
                aid = _AGENT_TILE_TO_ID.get(eid)
                true = true_pos.get(aid)
                if true is not None:
                    err = abs(est_pos[0] - true[0]) + abs(est_pos[1] - true[1])
                    tag = "exact" if err == 0 else f"err={err}"
                    parts.append(f"A{aid}: est{est_pos} true{true} [{tag}]")
                else:
                    parts.append(f"A{aid}: est{est_pos} [dead/off-board]")
            print(f"           Tracker: {' | '.join(parts)}")


def _make_mcts_agent(agent_id, strategy, args):
    return MCTSPOAgent(
        agent_id=agent_id,
        po_strategy=strategy,
        num_determinizations=args.num_determinizations,
        c_puct=args.c_puct,
        env_id=args.env,
    )


# Team seat assignments (Pommerman team competition layout)
_TEAM0_IDS = [0, 2]
_TEAM1_IDS = [1, 3]


def runner(runner_id, num_episodes, fifo, args):
    """
    Run num_episodes games. Three modes (set by flags):

    FFA (default):
        One MCTS-PO agent at seat (runner_id % 4) vs 3 SimpleAgents.

    --team_mcts:
        Team 0 (A0+A2) = MCTS-PO  vs  Team 1 (A1+A3) = SimpleAgent.

    --vs_mcts:
        Team 0 (A0+A2) = MCTS-PO (--po_strategy)
        vs
        Team 1 (A1+A3) = MCTS-PO (--opp_strategy, defaults to same).
        Both teams' rewards are reported.
    """
    is_team = 'Team' in args.env
    is_team_mcts = args.team_mcts and is_team
    is_vs_mcts   = args.vs_mcts   and is_team

    # All 4 env slots are SimpleAgents so env.act() always produces valid actions.
    # We override whichever seats are MCTS-controlled below.
    agent_list = [SimpleAgent() for _ in range(NUM_AGENTS)]
    env = pommerman.make(args.env, agent_list)

    # Build {agent_id: MCTSPOAgent} for every MCTS-controlled seat
    if is_vs_mcts:
        opp_strategy = args.opp_strategy or args.po_strategy
        mcts_agents = {
            0: _make_mcts_agent(0, args.po_strategy, args),
            2: _make_mcts_agent(2, args.po_strategy, args),
            1: _make_mcts_agent(1, opp_strategy, args),
            3: _make_mcts_agent(3, opp_strategy, args),
        }
        team0_ids = _TEAM0_IDS
        team1_ids = _TEAM1_IDS
    elif is_team_mcts:
        mcts_agents = {
            0: _make_mcts_agent(0, args.po_strategy, args),
            2: _make_mcts_agent(2, args.po_strategy, args),
        }
        team0_ids = _TEAM0_IDS
        team1_ids = []
    else:
        agent_id = runner_id % NUM_AGENTS
        mcts_agents = {agent_id: _make_mcts_agent(agent_id, args.po_strategy, args)}
        team0_ids = [agent_id]
        team1_ids = []

    total_reward_t0 = 0.0
    total_reward_t1 = 0.0
    total_length    = 0
    ep_rewards: list = []   # per-episode reward for team0
    ep_lengths: list = []   # per-episode length

    for ep in range(num_episodes):
        obs = env.reset()
        for ag in mcts_agents.values():
            ag.reset()
        done   = False
        length = 0
        ep_start = time.time()

        if args.log_steps:
            if is_vs_mcts:
                opp_strategy = args.opp_strategy or args.po_strategy
                label = (f"Team0 (A0+A2, {args.po_strategy}) "
                         f"vs Team1 (A1+A3, {opp_strategy})")
            elif is_team_mcts:
                label = f"Team0 MCTS (A0+A2, {args.po_strategy}) vs Team1 Simple (A1+A3)"
            else:
                label = f"Agent {team0_ids[0]} ({args.po_strategy})"
            print(f"\n=== Episode {ep+1} | Runner {runner_id} | {label} ===")

        while not done:
            root_state = env.get_json_info()

            for aid, ag in mcts_agents.items():
                ag.update_tracker(obs[aid])

            # Start from SimpleAgent baseline actions, then override MCTS seats
            actions = env.act(obs)
            for aid, ag in mcts_agents.items():
                action, _ = ag.search(
                    root_state, num_iters=args.mcts_iters, temperature=args.temperature
                )
                actions[aid] = action

            if args.log_steps:
                for aid, ag in mcts_agents.items():
                    strategy = (args.po_strategy if aid in _TEAM0_IDS
                                else (args.opp_strategy or args.po_strategy))
                    _log_step(length + 1, aid, strategy,
                              actions[aid], actions, root_state, ag)

            obs, rewards, done, info = env.step(actions)
            length += 1

            if args.render:
                env.render()

        ep_time = time.time() - ep_start

        reward_t0 = sum(rewards[i] for i in team0_ids) / len(team0_ids)
        reward_t1 = (sum(rewards[i] for i in team1_ids) / len(team1_ids)
                     if team1_ids else None)
        total_reward_t0 += reward_t0
        if reward_t1 is not None:
            total_reward_t1 += reward_t1
        total_length += length
        ep_rewards.append(reward_t0)
        ep_lengths.append(length)

        if args.verbose or num_episodes <= 5:
            if is_vs_mcts:
                print(f"Runner {runner_id} | Ep {ep+1}/{num_episodes} | "
                      f"T0(MCTS): {reward_t0:.2f}  T1(MCTS-opp): {reward_t1:.2f} | "
                      f"Length: {length} | Time: {ep_time:.1f}s")
            elif is_team_mcts:
                print(f"Runner {runner_id} | Ep {ep+1}/{num_episodes} | "
                      f"T0(MCTS): {reward_t0:.2f} | "
                      f"Length: {length} | Time: {ep_time:.1f}s")
            else:
                aid = team0_ids[0]
                print(f"Runner {runner_id} | Ep {ep+1}/{num_episodes} | "
                      f"Reward: {rewards[aid]} | Length: {length} | "
                      f"Time: {ep_time:.1f}s | "
                      f"Step: {ep_time/max(length,1)*1000:.0f}ms/step")

    env.close()

    fifo.send({
        'runner_id':      runner_id,
        'avg_reward':     total_reward_t0 / num_episodes,
        'avg_reward_opp': (total_reward_t1 / num_episodes if team1_ids else None),
        'avg_length':     total_length / num_episodes,
        'num_episodes':   num_episodes,
        'ep_rewards':     ep_rewards,
        'ep_lengths':     ep_lengths,
    })


def main():
    parser = argparse.ArgumentParser(
        description='MCTS with Partial Observability for Pommerman')

    # PO strategy
    parser.add_argument('--po_strategy', type=str, default='naive',
                        choices=['naive', 'estimation', 'particle'],
                        help='Partial observability handling strategy')

    # MCTS parameters
    parser.add_argument('--mcts_iters', type=int, default=100,
                        help='MCTS iterations per step')
    parser.add_argument('--c_puct', type=float, default=1.0,
                        help='UCB exploration constant')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Action selection temperature (0=greedy)')
    parser.add_argument('--num_determinizations', type=int, default=5,
                        help='Determinizations for particle IS-MCTS')

    # Evaluation
    parser.add_argument('--num_episodes', type=int, default=40,
                        help='Total episodes to run')
    parser.add_argument('--num_runners', type=int, default=4,
                        help='Parallel runner processes')
    parser.add_argument('--env', type=str, default='PommeFFACompetition-v0',
                        help='Environment ID')
    parser.add_argument('--team_mcts', action='store_true',
                        help='Team mode: MCTS-PO team (A0+A2) vs SimpleAgent team (A1+A3)')
    parser.add_argument('--vs_mcts', action='store_true',
                        help='Team mode: MCTS-PO team (A0+A2) vs MCTS-PO team (A1+A3)')
    parser.add_argument('--opp_strategy', type=str, default=None,
                        choices=['naive', 'estimation', 'particle'],
                        help='PO strategy for the opposing MCTS team (--vs_mcts only). '
                             'Defaults to --po_strategy if unset.')
    parser.add_argument('--render', action='store_true',
                        help='Render game (only with 1 runner)')
    parser.add_argument('--log_steps', action='store_true',
                        help='Print per-step log: positions, actions, tracker estimates vs truth')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    assert args.num_episodes % args.num_runners == 0, \
        "num_episodes must be divisible by num_runners"

    episodes_per_runner = args.num_episodes // args.num_runners

    is_team = 'Team' in args.env
    opp_strategy = args.opp_strategy or args.po_strategy
    print(f"{'='*60}")
    print(f"MCTS-PO Evaluation")
    print(f"  Strategy:        {args.po_strategy}")
    print(f"  MCTS iterations: {args.mcts_iters}")
    print(f"  Episodes:        {args.num_episodes} ({args.num_runners} runners)")
    print(f"  Environment:     {args.env}")
    if is_team and args.vs_mcts:
        print(f"  Mode:            2v2 MCTS vs MCTS")
        print(f"    Team 0 (A0+A2): {args.po_strategy}")
        print(f"    Team 1 (A1+A3): {opp_strategy}")
    elif is_team and args.team_mcts:
        print(f"  Mode:            2v2 Team — MCTS team (A0+A2) vs SimpleAgent team (A1+A3)")
    elif is_team:
        print(f"  Mode:            2v2 Team (single MCTS agent + SimpleAgent teammate)")
    if args.po_strategy == 'particle':
        print(f"  Determinizations: {args.num_determinizations}")
    print(f"{'='*60}")

    start_time = time.time()

    if args.num_runners == 1:
        # Single process — easier to debug
        parent_conn, child_conn = mp.Pipe()
        runner(0, episodes_per_runner, child_conn, args)
        results = [parent_conn.recv()]
    else:
        # Multiprocess
        ctx = mp.get_context('spawn')
        pipes = []
        processes = []
        for i in range(args.num_runners):
            parent_conn, child_conn = ctx.Pipe()
            pipes.append(parent_conn)
            p = ctx.Process(target=runner,
                            args=(i, episodes_per_runner, child_conn, args))
            processes.append(p)
            p.start()

        results = [pipe.recv() for pipe in pipes]
        for p in processes:
            p.join()

    # Aggregate results
    total_episodes  = 0
    weighted_reward = 0.0
    weighted_opp    = 0.0
    weighted_length = 0.0
    all_ep_rewards: list = []
    all_ep_lengths: list = []
    has_opp = any(r['avg_reward_opp'] is not None for r in results)
    for r in results:
        n = r['num_episodes']
        weighted_reward += r['avg_reward'] * n
        weighted_length += r['avg_length'] * n
        if r['avg_reward_opp'] is not None:
            weighted_opp += r['avg_reward_opp'] * n
        total_episodes += n
        all_ep_rewards.extend(r.get('ep_rewards', []))
        all_ep_lengths.extend(r.get('ep_lengths', []))

    overall_reward = weighted_reward / total_episodes
    overall_length = weighted_length / total_episodes
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"RESULTS")
    if has_opp:
        overall_opp = weighted_opp / total_episodes
        print(f"  Team 0 avg reward ({args.po_strategy}):  {overall_reward:.3f}")
        print(f"  Team 1 avg reward ({opp_strategy}):  {overall_opp:.3f}")
    else:
        print(f"  Avg reward:  {overall_reward:.3f}")
    print(f"  Avg length:  {overall_length:.0f}")
    print(f"  Total time:  {total_time:.1f}s")
    print(f"  Episodes:    {total_episodes}")
    print(f"{'='*60}")

    # Emit machine-readable line for compare_strategies.py to parse
    import json as _json
    print(f"__DATA__ {_json.dumps({'ep_rewards': all_ep_rewards, 'ep_lengths': all_ep_lengths})}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Compare all three PO strategies side by side.

Runs each strategy for the same number of episodes and produces
a table and plots suitable for the final project report.

Usage:
    python mcts_po/compare_strategies.py --mcts_iters 75 --num_episodes 20
    python mcts_po/compare_strategies.py --quick   # fast smoke test
    python mcts_po/compare_strategies.py --plot_only results.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time


def run_strategy(strategy, mcts_iters, num_episodes, num_runners, env,
                 team_mcts=False, num_det=5):
    """Run one strategy and parse output, including per-episode data."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_mcts_po.py')
    cmd = [
        sys.executable, script,
        '--po_strategy', strategy,
        '--mcts_iters', str(mcts_iters),
        '--num_episodes', str(num_episodes),
        '--num_runners', str(num_runners),
        '--env', env,
        '--num_determinizations', str(num_det),
    ]
    if team_mcts:
        cmd.append('--team_mcts')

    print(f"\n--- Running: {strategy} {'(team_mcts)' if team_mcts else ''} ---")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    elapsed = time.time() - start

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])

    avg_reward, avg_length = None, None
    ep_rewards, ep_lengths = [], []

    for line in result.stdout.split('\n'):
        if 'Avg reward:' in line:
            m = re.search(r'Avg reward:\s+([-\d.]+)', line)
            if m:
                avg_reward = float(m.group(1))
        if 'Avg length:' in line:
            m = re.search(r'Avg length:\s+([\d.]+)', line)
            if m:
                avg_length = float(m.group(1))
        if line.startswith('__DATA__ '):
            try:
                data = json.loads(line[len('__DATA__ '):])
                ep_rewards = data.get('ep_rewards', [])
                ep_lengths = data.get('ep_lengths', [])
            except json.JSONDecodeError:
                pass

    return {
        'strategy': strategy,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'ep_rewards': ep_rewards,
        'ep_lengths': ep_lengths,
        'time': elapsed,
    }


def plot_results(results, output_path=None):
    """Generate a 2x2 comparison figure from collected results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plots. Run: pip install matplotlib")
        return

    strategies = [r['strategy'] for r in results]
    colors = {'naive': '#e74c3c', 'estimation': '#3498db', 'particle': '#2ecc71'}
    clrs = [colors.get(s, '#95a5a6') for s in strategies]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('MCTS-PO Strategy Comparison', fontsize=14, fontweight='bold')

    # ── 1. Average reward bar chart with std error bars ──────────────────────
    ax = axes[0, 0]
    avgs = [r['avg_reward'] if r['avg_reward'] is not None else 0.0 for r in results]
    stds = [np.std(r['ep_rewards']) / max(len(r['ep_rewards']) ** 0.5, 1)
            for r in results]
    bars = ax.bar(strategies, avgs, color=clrs, edgecolor='black', linewidth=0.8)
    ax.errorbar(strategies, avgs, yerr=stds, fmt='none', color='black',
                capsize=5, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title('Average Reward (±SE)')
    ax.set_ylabel('Reward')
    ax.set_ylim(min(avgs) - 0.3, max(avgs) + 0.3)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # ── 2. Reward distribution (box plot) ────────────────────────────────────
    ax = axes[0, 1]
    ep_data = [r['ep_rewards'] if r['ep_rewards'] else [0.0] for r in results]
    bp = ax.boxplot(ep_data, labels=strategies, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    for patch, clr in zip(bp['boxes'], clrs):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title('Reward Distribution per Episode')
    ax.set_ylabel('Reward')

    # ── 3. Win / Draw / Loss breakdown ───────────────────────────────────────
    ax = axes[1, 0]
    x = np.arange(len(strategies))
    width = 0.25
    for i, (label, val, fc) in enumerate([('Win (+1)', 1, '#27ae60'),
                                           ('Draw (0)', 0, '#f39c12'),
                                           ('Loss (-1)', -1, '#e74c3c')]):
        counts = [
            sum(1 for ep in r['ep_rewards'] if ep == val) / max(len(r['ep_rewards']), 1)
            for r in results
        ]
        ax.bar(x + (i - 1) * width, counts, width, label=label,
               color=fc, edgecolor='black', linewidth=0.7, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_title('Win / Draw / Loss Rate')
    ax.set_ylabel('Fraction of Episodes')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # ── 4. Survival length CDF ───────────────────────────────────────────────
    ax = axes[1, 1]
    for r, clr in zip(results, clrs):
        lengths = sorted(r['ep_lengths']) if r['ep_lengths'] else [0]
        cdf = np.arange(1, len(lengths) + 1) / len(lengths)
        ax.step(lengths, cdf, where='post', color=clr,
                linewidth=2, label=r['strategy'])
    ax.set_title('Survival Length CDF')
    ax.set_xlabel('Steps survived')
    ax.set_ylabel('Cumulative fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'strategy_comparison.png'
        )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mcts_iters', type=int, default=75)
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--num_runners', type=int, default=4)
    parser.add_argument('--env', type=str, default='PommeTeamCompetition-v0')
    parser.add_argument('--team_mcts', action='store_true',
                        help='Use --team_mcts mode (both A0+A2 are MCTS)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test: 8 episodes, 50 iters')
    parser.add_argument('--save', type=str, default=None,
                        help='Save raw results JSON to this path')
    parser.add_argument('--plot_only', type=str, default=None,
                        help='Skip running — load results from this JSON and plot')
    parser.add_argument('--plot_out', type=str, default=None,
                        help='Output path for the comparison PNG')
    args = parser.parse_args()

    if args.plot_only:
        with open(args.plot_only) as f:
            results = json.load(f)
        plot_results(results, args.plot_out)
        return

    if args.quick:
        args.num_episodes = 8
        args.num_runners = 4
        args.mcts_iters = 50

    assert args.num_episodes % args.num_runners == 0, \
        "num_episodes must be divisible by num_runners"

    strategies = ['naive', 'estimation', 'particle']
    results = []

    print("=" * 70)
    print("MCTS-PO STRATEGY COMPARISON")
    print(f"  MCTS iters:  {args.mcts_iters}")
    print(f"  Episodes:    {args.num_episodes} ({args.num_runners} runners each)")
    print(f"  Env:         {args.env}")
    print(f"  Team MCTS:   {args.team_mcts}")
    print("=" * 70)

    for strat in strategies:
        r = run_strategy(
            strat, args.mcts_iters, args.num_episodes,
            args.num_runners, args.env,
            team_mcts=args.team_mcts,
        )
        results.append(r)

    # Print table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Avg Reward':<12} {'Win%':<8} {'Avg Length':<12} {'Time (s)'}")
    print("-" * 60)
    for r in results:
        rw   = f"{r['avg_reward']:.3f}"  if r['avg_reward']  is not None else "N/A"
        ln   = f"{r['avg_length']:.0f}"  if r['avg_length']  is not None else "N/A"
        wins = (sum(1 for x in r['ep_rewards'] if x == 1) / max(len(r['ep_rewards']), 1)
                * 100)
        print(f"{r['strategy']:<15} {rw:<12} {wins:<8.1f} {ln:<12} {r['time']:.1f}")

    print(f"\n(Reward: 1=win, 0=tie/timeout, -1=loss)")

    # Save raw results
    save_path = args.save or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'comparison_results.json'
    )
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to: {save_path}")

    # Generate plots
    plot_results(results, args.plot_out)


if __name__ == '__main__':
    main()

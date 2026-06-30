"""
training.py — TD(0) Self-Play Training & Evaluation (UNCHANGED LOGIC)
=========================================================================
Direct extraction of run_self_play_training() and evaluate_agent() from the
original Streamlit app. The TD(0) update rule, epsilon-greedy exploration
schedule, and evaluation match logic are byte-for-byte the same.

ONLY CHANGE: the original functions accepted Streamlit widget objects
(`progress_bar`, `status_text`) and called `.progress()` / `.text()` on them
directly. Since Pygame has no equivalent widgets, these are now plain
optional callables:
    - progress_callback(fraction: float) -> None
    - status_callback(message: str) -> None
This preserves the exact same call sites/frequency, just swapping
`progress_bar.progress(x)` for `progress_callback(x)` and
`status_text.text(s)` for `status_callback(s)`. No training math changed.
"""

from __future__ import annotations

import math
from typing import Optional, List, Dict, Callable

from game import (
    initial_board,
    get_all_legal_moves,
    apply_move,
    canonical_view,
    extract_features,
    DEFAULT_WEIGHTS,
)
from ai import CheckersAgent, RandomAgent, GreedyCaptureAgent


def run_self_play_training(
    episodes: int,
    train_depth: int = 2,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
    alpha: float = 0.01,
    gamma: float = 0.98,
    weights: Optional[List[float]] = None,
    max_plies: int = 120,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> List[float]:
    weights = list(weights) if weights is not None else list(DEFAULT_WEIGHTS)
    agent = CheckersAgent(weights=weights, depth=train_depth)
    epsilon = epsilon_start
    decay = (epsilon_end / epsilon_start) ** (1.0 / max(episodes, 1))

    for ep in range(1, episodes + 1):
        board = initial_board()
        player = 1
        plies = 0

        while True:
            moves = get_all_legal_moves(board, player)
            if not moves or plies >= max_plies:
                break

            feats_t = extract_features(canonical_view(board, player))
            v_t = sum(w * f for w, f in zip(agent.weights, feats_t))

            move, _, _ = agent.select_action(board, player, depth=train_depth, epsilon=epsilon)
            new_board = apply_move(board, move)
            plies += 1

            opp_moves = get_all_legal_moves(new_board, -player)
            if not opp_moves:
                target = 1.0
                terminal = True
            elif plies >= max_plies:
                target = 0.0
                terminal = True
            else:
                feats_next = extract_features(canonical_view(new_board, -player))
                v_next = sum(w * f for w, f in zip(agent.weights, feats_next))
                target = -gamma * v_next
                terminal = False

            error = target - v_t
            for i in range(len(agent.weights)):
                agent.weights[i] += alpha * error * feats_t[i]

            board = new_board
            player = -player
            if terminal:
                break

        epsilon = max(epsilon_end, epsilon * decay)

        if progress_callback and ep % max(1, episodes // 200) == 0:
            progress_callback(ep / episodes)
        if status_callback and ep % max(1, episodes // 50) == 0:
            wnorm = math.sqrt(sum(w * w for w in agent.weights))
            status_callback(f"Game {ep:,}/{episodes:,} | ε={epsilon:.3f} | ‖w‖={wnorm:.2f}")

    return agent.weights


def evaluate_agent(agent: CheckersAgent, n_games: int = 40, eval_depth: Optional[int] = None) -> Dict[str, Dict]:
    opponents = {"RandomAgent": RandomAgent(), "GreedyAgent": GreedyCaptureAgent()}
    results: Dict[str, Dict] = {}
    for name, opp in opponents.items():
        wins = draws = losses = 0
        for g in range(n_games):
            agent_side = 1 if g % 2 == 0 else -1
            board = initial_board()
            player = 1
            plies = 0
            winner = None
            while True:
                moves = get_all_legal_moves(board, player)
                if not moves:
                    winner = -player
                    break
                if plies >= 150:
                    winner = 0
                    break
                if player == agent_side:
                    move, _, _ = agent.select_action(board, player, depth=eval_depth or agent.depth, epsilon=0.0)
                else:
                    move, _, _ = opp.select_action(board, player)
                board = apply_move(board, move)
                plies += 1
                player = -player
            if winner == agent_side:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
        results[name] = {
            "wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / n_games, "non_loss_rate": (wins + draws) / n_games,
        }
    return results


DIFFICULTY_CONFIG = {
    "Easy":   {"path": "models/checkers_easy.pkl",   "episodes": 1000,  "train_depth": 1, "play_depth": 2, "eps_start": 0.35, "eps_end": 0.15, "play_epsilon": 0.35},
    "Medium": {"path": "models/checkers_medium.pkl", "episodes": 5000,  "train_depth": 2, "play_depth": 3, "eps_start": 0.25, "eps_end": 0.05, "play_epsilon": 0.10},
    "Hard":   {"path": "models/checkers_hard.pkl",   "episodes": 12000, "train_depth": 2, "play_depth": 4, "eps_start": 0.20, "eps_end": 0.02, "play_epsilon": 0.0},
}
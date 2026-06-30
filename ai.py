"""
ai.py — Agents & Search (UNCHANGED LOGIC)
===========================================
Direct extraction of CheckersAgent, RandomAgent, GreedyCaptureAgent, and the
negamax + alpha-beta search from the original Streamlit app. No algorithmic
changes — only the import of board/rules functions now comes from game.py
instead of being defined inline, and the Streamlit/joblib save-path handling
is untouched.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import joblib

from game import (
    get_all_legal_moves,
    apply_move,
    canonical_view,
    extract_features,
    DEFAULT_WEIGHTS,
    WIN_SCORE,
)


class CheckersAgent:
    def __init__(self, weights: Optional[List[float]] = None, depth: int = 3, label: str = "untrained"):
        self.weights = list(weights) if weights is not None else list(DEFAULT_WEIGHTS)
        self.depth = depth
        self.label = label
        self.episodes_trained: Optional[int] = None

    def evaluate(self, board: List[List[int]], player: int) -> float:
        feats = extract_features(canonical_view(board, player))
        return sum(w * f for w, f in zip(self.weights, feats))

    def negamax(self, board: List[List[int]], player: int, depth: int, alpha: float, beta: float) -> float:
        moves = get_all_legal_moves(board, player)
        if not moves:
            return -WIN_SCORE
        if depth == 0:
            return self.evaluate(board, player)
        best = -math.inf
        random.shuffle(moves)
        for move in moves:
            new_board = apply_move(board, move)
            val = -self.negamax(new_board, -player, depth - 1, -beta, -alpha)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def select_action(
        self, board: List[List[int]], player: int,
        depth: Optional[int] = None, epsilon: float = 0.0,
    ) -> Tuple[Optional[Dict], List[Tuple[Dict, float]], Optional[float]]:
        moves = get_all_legal_moves(board, player)
        if not moves:
            return None, [], None
        d = depth if depth is not None else self.depth

        if epsilon > 0 and random.random() < epsilon:
            return random.choice(moves), [], None

        scored = []
        for move in moves:
            new_board = apply_move(board, move)
            val = -self.negamax(new_board, -player, d - 1, -math.inf, math.inf)
            scored.append((move, val))
        scored.sort(key=lambda x: x[1], reverse=True)
        best_val = scored[0][1]
        best_moves = [m for m, v in scored if abs(v - best_val) < 1e-9]
        chosen = random.choice(best_moves)
        return chosen, scored[:3], best_val

    def save(self, path: str, episodes_trained: Optional[int] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "weights": self.weights,
            "depth": self.depth,
            "label": self.label,
            "episodes_trained": episodes_trained,
        }, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "CheckersAgent":
        data = joblib.load(path)
        agent = cls(weights=data["weights"], depth=data.get("depth", 3), label=data.get("label", "loaded"))
        agent.episodes_trained = data.get("episodes_trained")
        return agent


class RandomAgent:
    label = "Random"

    def select_action(self, board, player, **kwargs):
        moves = get_all_legal_moves(board, player)
        if not moves:
            return None, [], None
        return random.choice(moves), [], None


class GreedyCaptureAgent:
    label = "Greedy"

    def select_action(self, board, player, **kwargs):
        moves = get_all_legal_moves(board, player)
        if not moves:
            return None, [], None

        def score(move):
            s = len(move["captured"]) * 10
            r0, c0 = move["path"][0]
            rF, cF = move["path"][-1]
            promoted = abs(board[r0][c0]) == 1 and abs(apply_move(board, move)[rF][cF]) == 2
            if promoted:
                s += 5
            s += (7 - rF) if player == 1 else rF
            return s

        best = max(moves, key=score)
        return best, [], None
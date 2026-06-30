"""
game.py — Core Checkers Rules Engine
=====================================
UNCHANGED LOGIC. This is a direct extraction of the board representation,
move generation, capture-chain search, and move application from the
original Streamlit app. No rules, no algorithms, and no behavior have been
modified — only the Streamlit import/UI coupling has been removed, since
none of this code actually depended on Streamlit at runtime.
"""

from __future__ import annotations

from typing import List, Dict, Tuple

MAN_DIRS = {1: [(-1, -1), (-1, 1)], -1: [(1, -1), (1, 1)]}
KING_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
WIN_SCORE = 1000.0

N_FEATURES = 10
DEFAULT_WEIGHTS = [0.0, 1.0, 1.7, 0.05, 0.10, 0.05, -0.04, 0.03, 0.05, -0.05]


def initial_board() -> List[List[int]]:
    board = [[0] * 8 for _ in range(8)]
    for r in range(3):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r][c] = -1
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                board[r][c] = 1
    return board


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def board_copy(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]


def get_simple_moves(board: List[List[int]], r: int, c: int) -> List[Dict]:
    v = board[r][c]
    owner = 1 if v > 0 else -1
    dirs = KING_DIRS if abs(v) == 2 else MAN_DIRS[owner]
    moves = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc) and board[nr][nc] == 0:
            moves.append({"path": [(r, c), (nr, nc)], "captured": [], "is_capture": False})
    return moves


def find_capture_sequences(board: List[List[int]], r: int, c: int) -> List[Dict]:
    """All maximal capture chains starting at (r, c)."""
    piece = board[r][c]
    owner = 1 if piece > 0 else -1
    results: List[Dict] = []

    def recurse(rr: int, cc: int, state: List[List[int]], path: List[Tuple[int, int]],
                captured: List[Tuple[int, int]]):
        dirs_here = KING_DIRS if abs(state[rr][cc]) == 2 else MAN_DIRS[owner]
        extended = False
        for dr, dc in dirs_here:
            mr, mc = rr + dr, cc + dc
            lr, lc = rr + 2 * dr, cc + 2 * dc
            if not (in_bounds(mr, mc) and in_bounds(lr, lc)):
                continue
            mid_val = state[mr][mc]
            if mid_val == 0 or (mid_val > 0) == (owner > 0):
                continue
            if state[lr][lc] != 0:
                continue
            new_state = board_copy(state)
            moving_piece = new_state[rr][cc]
            new_state[rr][cc] = 0
            new_state[mr][mc] = 0
            new_state[lr][lc] = moving_piece
            extended = True
            recurse(lr, lc, new_state, path + [(lr, lc)], captured + [(mr, mc)])
        if not extended and captured:
            results.append({"path": path, "captured": captured})

    recurse(r, c, board, [(r, c)], [])
    return results


def get_all_legal_moves(board: List[List[int]], player: int) -> List[Dict]:
    capture_moves: List[Dict] = []
    for r in range(8):
        for c in range(8):
            v = board[r][c]
            if v == 0 or (v > 0) != (player > 0):
                continue
            for seq in find_capture_sequences(board, r, c):
                capture_moves.append({"path": seq["path"], "captured": seq["captured"], "is_capture": True})
    if capture_moves:
        return capture_moves
    simple_moves: List[Dict] = []
    for r in range(8):
        for c in range(8):
            v = board[r][c]
            if v == 0 or (v > 0) != (player > 0):
                continue
            simple_moves.extend(get_simple_moves(board, r, c))
    return simple_moves


def apply_move(board: List[List[int]], move: Dict) -> List[List[int]]:
    new_board = board_copy(board)
    (r0, c0) = move["path"][0]
    piece = new_board[r0][c0]
    new_board[r0][c0] = 0
    for (cr, cc) in move["captured"]:
        new_board[cr][cc] = 0
    rF, cF = move["path"][-1]
    new_board[rF][cF] = piece
    owner = 1 if piece > 0 else -1
    if abs(piece) == 1:
        if (owner == 1 and rF == 0) or (owner == -1 and rF == 7):
            new_board[rF][cF] = 2 * owner
    return new_board


def canonical_view(board: List[List[int]], player: int) -> List[List[int]]:
    if player == 1:
        return board
    return [[-board[7 - r][7 - c] for c in range(8)] for r in range(8)]


def extract_features(board: List[List[int]]) -> List[float]:
    my_men = opp_men = my_kings = opp_kings = 0
    my_advance = opp_advance = 0
    my_back = opp_back = 0
    my_center = opp_center = 0
    my_edge = opp_edge = 0

    for r in range(8):
        for c in range(8):
            v = board[r][c]
            if v == 0:
                continue
            mine = v > 0
            king = abs(v) == 2
            if mine:
                if king:
                    my_kings += 1
                else:
                    my_men += 1
                    my_advance += (7 - r)
                if r == 7:
                    my_back += 1
                if 2 <= r <= 5 and 2 <= c <= 5:
                    my_center += 1
                if c in (0, 7):
                    my_edge += 1
            else:
                if king:
                    opp_kings += 1
                else:
                    opp_men += 1
                    opp_advance += r
                if r == 0:
                    opp_back += 1
                if 2 <= r <= 5 and 2 <= c <= 5:
                    opp_center += 1
                if c in (0, 7):
                    opp_edge += 1

    my_moves = get_all_legal_moves(board, 1)
    opp_moves = get_all_legal_moves(board, -1)
    my_forced_capture = 1.0 if (my_moves and my_moves[0]["is_capture"]) else 0.0
    opp_forced_capture = 1.0 if (opp_moves and opp_moves[0]["is_capture"]) else 0.0
    mobility_diff = (len(my_moves) - len(opp_moves)) / 10.0

    return [
        1.0,
        float(my_men - opp_men),
        float(my_kings - opp_kings),
        (my_advance - opp_advance) / 12.0,
        float(my_back - opp_back),
        float(my_center - opp_center),
        float(my_edge - opp_edge),
        mobility_diff,
        my_forced_capture,
        opp_forced_capture,
    ]
"""
ui.py — Event Handling & Game State Management
==================================================
Owns the mutable UI/game state (selected square, legal destinations,
captured counts, difficulty, game-over flag) and translates Pygame events
into calls against the UNCHANGED rules engine (game.py) and agents (ai.py).
"""

import pygame
import time

from game import initial_board, get_all_legal_moves, apply_move
from ai import CheckersAgent, RandomAgent, GreedyCaptureAgent
from training import DIFFICULTY_CONFIG
from renderer import Renderer, SIDEBAR_X, BOARD_MARGIN_Y, BOARD_PIXELS, SQ_PIXELS

DIFF_KEYS = list(DIFFICULTY_CONFIG.keys())


class GameState:
    def __init__(self):
        self.board = initial_board()
        self.player = 1
        self.human_side = 1
        self.difficulty = "Medium"
        self.selected = None
        self.dest_map = {}
        self.done = False
        self.winner = None
        self.last_move = None
        self.captured_red = 0
        self.captured_black = 0
        self.ply_count = 0
        self.ai_thinking = False
        self._agent_cache = {}

    def reset(self):
        self.board = initial_board()
        self.player = 1
        self.selected = None
        self.dest_map = {}
        self.done = False
        self.winner = None
        self.last_move = None
        self.captured_red = 0
        self.captured_black = 0
        self.ply_count = 0
        self.ai_thinking = False

    def get_agent(self):
        cfg = DIFFICULTY_CONFIG[self.difficulty]
        if self.difficulty not in self._agent_cache:
            try:
                self._agent_cache[self.difficulty] = CheckersAgent.load(cfg["path"])
            except FileNotFoundError:
                self._agent_cache[self.difficulty] = CheckersAgent(depth=cfg["play_depth"])
        return self._agent_cache[self.difficulty]

    def apply_ui_move(self, move):
        captured_count = len(move["captured"])
        if captured_count:
            if self.player == 1:
                self.captured_black += captured_count
            else:
                self.captured_red += captured_count

        self.board = apply_move(self.board, move)
        self.last_move = move
        self.ply_count += 1

        next_player = -self.player
        opp_moves = get_all_legal_moves(self.board, next_player)
        if not opp_moves:
            self.done = True
            self.winner = self.player
        elif self.ply_count >= 300:
            self.done = True
            self.winner = 0
        else:
            self.player = next_player

        self.selected = None
        self.dest_map = {}

    def handle_square_click(self, r, c):
        if self.done or self.player != self.human_side or self.ai_thinking:
            return

        v = self.board[r][c]

        if self.selected == (r, c):
            self.selected = None
            self.dest_map = {}
            return

        if self.selected is not None and (r, c) in self.dest_map:
            self.apply_ui_move(self.dest_map[(r, c)])
            return

        if v != 0 and (v > 0) == (self.player > 0):
            moves = [m for m in get_all_legal_moves(self.board, self.player) if m["path"][0] == (r, c)]
            if moves:
                self.selected = (r, c)
                self.dest_map = {m["path"][-1]: m for m in moves}
            else:
                self.selected = None
                self.dest_map = {}

    def do_ai_move(self):
        agent = self.get_agent()
        cfg = DIFFICULTY_CONFIG[self.difficulty]
        move, _, _ = agent.select_action(
            self.board, self.player,
            depth=cfg["play_depth"],
            epsilon=cfg.get("play_epsilon", 0.0),
        )
        if move is None:
            self.done = True
            self.winner = -self.player
            return
        self.apply_ui_move(move)


class UIController:
    """Wires Pygame events to GameState + Renderer, owns sidebar buttons."""

    def __init__(self, screen, state: GameState, renderer: Renderer):
        self.screen = screen
        self.state = state
        self.renderer = renderer
        self.ai_move_pending_at = None

        self.restart_rect = pygame.Rect(SIDEBAR_X, BOARD_MARGIN_Y + 180, 200, 44)
        self.diff_rects = {
            name: pygame.Rect(SIDEBAR_X, BOARD_MARGIN_Y + 250 + i * 50, 200, 40)
            for i, name in enumerate(DIFF_KEYS)
        }
        self.side_toggle_rect = pygame.Rect(SIDEBAR_X, BOARD_MARGIN_Y + 250 + len(DIFF_KEYS) * 50 + 20, 200, 44)
        self.game_over_restart_rect = None

    def handle_event(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        pos = event.pos

        if self.state.done:
            if self.game_over_restart_rect and self.game_over_restart_rect.collidepoint(pos):
                self.state.reset()
            return

        if self.restart_rect.collidepoint(pos):
            self.state.reset()
            return

        for name, rect in self.diff_rects.items():
            if rect.collidepoint(pos):
                self.state.difficulty = name
                return

        if self.side_toggle_rect.collidepoint(pos):
            self.state.human_side *= -1
            self.state.reset()
            return

        sq = self.renderer.pixel_to_square(pos)
        if sq:
            self.state.handle_square_click(*sq)

    def update(self):
        if self.state.done:
            return
        if self.state.player != self.state.human_side:
            self.state.ai_thinking = True
            if self.ai_move_pending_at is None:
                self.ai_move_pending_at = time.time()
            elif time.time() - self.ai_move_pending_at > 0.35:
                self.state.do_ai_move()
                self.ai_move_pending_at = None
                self.state.ai_thinking = False
        else:
            self.state.ai_thinking = False
            self.ai_move_pending_at = None

    def draw_sidebar(self):
        r = self.renderer
        s = self.state
        r.draw_panel_bg()
        r.draw_title("Checkers")
        r.draw_turn_indicator(s.player, s.player == s.human_side, s.ai_thinking)
        r.draw_captured_counter(s.captured_red, s.captured_black)

        mouse_pos = pygame.mouse.get_pos()

        r.draw_button(self.restart_rect, "Restart", hovered=self.restart_rect.collidepoint(mouse_pos))

        r.draw_text("Difficulty:", (SIDEBAR_X, BOARD_MARGIN_Y + 220), font=r.font_body, color=(170, 155, 130))
        for name, rect in self.diff_rects.items():
            is_active = name == s.difficulty
            label = f"{'> ' if is_active else ''}{name}"
            r.draw_button(rect, label, hovered=rect.collidepoint(mouse_pos), font=r.font_small)

        side_label = "You: Red (bottom)" if s.human_side == 1 else "You: Black (top)"
        r.draw_button(self.side_toggle_rect, side_label, hovered=self.side_toggle_rect.collidepoint(mouse_pos), font=r.font_small)

        if s.done:
            if s.winner == s.human_side:
                msg = "You win!"
            elif s.winner == 0:
                msg = "Draw"
            else:
                msg = "AI wins"
            box_rect = r.draw_game_over_overlay(msg)
            self.game_over_restart_rect = pygame.Rect(box_rect.centerx - 90, box_rect.bottom - 60, 180, 44)
            r.draw_button(self.game_over_restart_rect, "Play Again", hovered=self.game_over_restart_rect.collidepoint(mouse_pos))
        else:
            self.game_over_restart_rect = None
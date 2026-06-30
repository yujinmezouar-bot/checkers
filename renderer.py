import sys

def resource_path(relative_path):
    """Resolve asset paths whether running as a normal script or as a
    PyInstaller-bundled exe (which extracts bundled files to a temp
    folder referenced by sys._MEIPASS)."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return relative_path


"""
renderer.py — Pygame Drawing Layer
=====================================
Pure rendering code. No game-rule logic lives here — it only reads board
state (List[List[int]]) and UI state and draws pixels. All game logic stays
in game.py / ai.py / training.py untouched.

Coordinate convention: board[row][col], row 0 = top (Black's home row),
row 7 = bottom (Red's home row) — matching the original Streamlit app.
"""

import pygame
import os

ASSET_DIR = resource_path("assets")

# ---------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------
WINDOW_W, WINDOW_H = 1100, 820
BOARD_MARGIN_X = 40
BOARD_MARGIN_Y = 40
BOARD_PIXELS = 640          # rendered board size on screen
SQ_PIXELS = BOARD_PIXELS // 8
SIDEBAR_X = BOARD_MARGIN_X + BOARD_PIXELS + 40

COLOR_BG_PANEL = (30, 22, 16)
COLOR_TEXT = (235, 222, 196)
COLOR_TEXT_DIM = (170, 155, 130)
COLOR_HILITE_SELECT = (59, 130, 246)
COLOR_HILITE_DEST = (34, 197, 94)
COLOR_BUTTON = (90, 56, 32)
COLOR_BUTTON_HOVER = (120, 76, 44)
COLOR_BUTTON_BORDER = (200, 160, 90)

FONT_NAME = None  # use pygame default; swap for a .ttf in assets/ for a nicer look


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.assets = self._load_assets()
        self.font_title = pygame.font.SysFont(FONT_NAME, 40, bold=True)
        self.font_h2 = pygame.font.SysFont(FONT_NAME, 26, bold=True)
        self.font_body = pygame.font.SysFont(FONT_NAME, 20)
        self.font_small = pygame.font.SysFont(FONT_NAME, 16)

    def _load_assets(self):
        def load(name, size=None):
            path = os.path.join(ASSET_DIR, name)
            img = pygame.image.load(path).convert_alpha()
            if size:
                img = pygame.transform.smoothscale(img, size)
            return img

        board_img = load("board.png", (BOARD_PIXELS + 96, BOARD_PIXELS + 96))
        piece_size = (int(SQ_PIXELS * 0.82), int(SQ_PIXELS * 0.82))
        return {
            "board": board_img,
            "red_piece": load("red_piece.png", piece_size),
            "black_piece": load("black_piece.png", piece_size),
            "red_king": load("red_king.png", piece_size),
            "black_king": load("black_king.png", piece_size),
            "background": load("background.png", (WINDOW_W, WINDOW_H)),
        }

    # -----------------------------------------------------------
    # Coordinate helpers
    # -----------------------------------------------------------
    def square_rect(self, r, c):
        x = BOARD_MARGIN_X + c * SQ_PIXELS
        y = BOARD_MARGIN_Y + r * SQ_PIXELS
        return pygame.Rect(x, y, SQ_PIXELS, SQ_PIXELS)

    def pixel_to_square(self, pos):
        x, y = pos
        col = (x - BOARD_MARGIN_X) // SQ_PIXELS
        row = (y - BOARD_MARGIN_Y) // SQ_PIXELS
        if 0 <= row < 8 and 0 <= col < 8:
            return int(row), int(col)
        return None

    def square_center(self, r, c):
        rect = self.square_rect(r, c)
        return rect.center

    # -----------------------------------------------------------
    # Background / board
    # -----------------------------------------------------------
    def draw_background(self):
        self.screen.blit(self.assets["background"], (0, 0))

    def draw_board_frame(self):
        # board.png already includes the wooden frame + 8x8 squares baked in,
        # offset by 48px so it lines up with our square grid
        self.screen.blit(self.assets["board"], (BOARD_MARGIN_X - 48, BOARD_MARGIN_Y - 48))

    def draw_highlights(self, selected, dest_squares):
        if selected:
            rect = self.square_rect(*selected)
            pygame.draw.rect(self.screen, COLOR_HILITE_SELECT, rect, width=4, border_radius=4)
        for sq in dest_squares:
            rect = self.square_rect(*sq)
            pygame.draw.rect(self.screen, COLOR_HILITE_DEST, rect, width=4, border_radius=4)
            # small dot in center for empty-destination clarity
            cx, cy = rect.center
            pygame.draw.circle(self.screen, COLOR_HILITE_DEST, (cx, cy), 8)

    def draw_last_move(self, last_move):
        if not last_move:
            return
        for sq in (last_move["path"][0], last_move["path"][-1]):
            rect = self.square_rect(*sq)
            pygame.draw.rect(self.screen, (255, 213, 74), rect, width=3, border_radius=4)

    # -----------------------------------------------------------
    # Pieces
    # -----------------------------------------------------------
    def piece_image(self, value):
        if value == 1:
            return self.assets["red_piece"]
        if value == -1:
            return self.assets["black_piece"]
        if value == 2:
            return self.assets["red_king"]
        if value == -2:
            return self.assets["black_king"]
        return None

    def draw_pieces(self, board, animating_piece=None):
        """
        animating_piece: optional dict {"value", "pos": (x, y)} drawn at a
        custom pixel position mid-slide instead of its board square — used
        for smooth movement. The square it's animating FROM/TO should be
        skipped by the caller (pass board with that square already emptied
        or filled depending on animation phase — see ui.py).
        """
        for r in range(8):
            for c in range(8):
                v = board[r][c]
                if v == 0:
                    continue
                img = self.piece_image(v)
                cx, cy = self.square_center(r, c)
                rect = img.get_rect(center=(cx, cy))
                self.screen.blit(img, rect)

        if animating_piece:
            img = self.piece_image(animating_piece["value"])
            rect = img.get_rect(center=animating_piece["pos"])
            self.screen.blit(img, rect)

    # -----------------------------------------------------------
    # Sidebar panel
    # -----------------------------------------------------------
    def draw_panel_bg(self):
        panel_rect = pygame.Rect(SIDEBAR_X - 20, BOARD_MARGIN_Y - 10, WINDOW_W - SIDEBAR_X, BOARD_PIXELS + 20)
        pygame.draw.rect(self.screen, COLOR_BG_PANEL, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_BUTTON_BORDER, panel_rect, width=2, border_radius=12)

    def draw_text(self, text, pos, font=None, color=COLOR_TEXT):
        font = font or self.font_body
        surf = font.render(text, True, color)
        self.screen.blit(surf, pos)
        return surf.get_height()

    def draw_title(self, text):
        self.draw_text(text, (SIDEBAR_X, BOARD_MARGIN_Y), font=self.font_title)

    def draw_turn_indicator(self, player, is_human_turn, ai_thinking):
        y = BOARD_MARGIN_Y + 70
        if ai_thinking:
            label = "AI is thinking..."
            color = (255, 213, 74)
        elif is_human_turn:
            label = "Your turn"
            color = COLOR_HILITE_DEST
        else:
            label = "Red's turn" if player == 1 else "Black's turn"
            color = COLOR_TEXT
        self.draw_text(label, (SIDEBAR_X, y), font=self.font_h2, color=color)

    def draw_captured_counter(self, red_captured, black_captured):
        y = BOARD_MARGIN_Y + 120
        self.draw_text(f"Captured — Red: {red_captured}   Black: {black_captured}",
                        (SIDEBAR_X, y), font=self.font_small, color=COLOR_TEXT_DIM)

    def draw_button(self, rect, label, hovered=False, font=None):
        color = COLOR_BUTTON_HOVER if hovered else COLOR_BUTTON
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BUTTON_BORDER, rect, width=2, border_radius=8)
        font = font or self.font_body
        surf = font.render(label, True, COLOR_TEXT)
        text_rect = surf.get_rect(center=rect.center)
        self.screen.blit(surf, text_rect)

    def draw_game_over_overlay(self, message):
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        box_w, box_h = 480, 220
        box_rect = pygame.Rect((WINDOW_W - box_w) // 2, (WINDOW_H - box_h) // 2, box_w, box_h)
        pygame.draw.rect(self.screen, COLOR_BG_PANEL, box_rect, border_radius=16)
        pygame.draw.rect(self.screen, (255, 213, 74), box_rect, width=3, border_radius=16)

        surf = self.font_title.render(message, True, COLOR_TEXT)
        text_rect = surf.get_rect(center=(box_rect.centerx, box_rect.centery - 30))
        self.screen.blit(surf, text_rect)

        return box_rect  # caller draws the "Restart" button below this
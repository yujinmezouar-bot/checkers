"""
main.py — Entry Point
========================
Initializes Pygame, creates the window, and runs the main loop:
    1. Poll events -> UIController.handle_event()
    2. Update AI turn timer -> UIController.update()
    3. Draw background, board, pieces (with animation), sidebar
    4. Flip display

Smooth movement: when a move is applied, instead of snapping the piece
instantly, we interpolate its pixel position from the source square to the
destination square over a short duration. This is purely visual — the
actual board state (game.py) is already updated immediately when the move
is applied; we just delay *drawing* the piece at its final square until the
animation finishes.

Animation trigger: we watch state.ply_count, which only increments inside
GameState.apply_ui_move(). Comparing it before/after each event or update
call tells us precisely when a move just landed, without relying on
fragile object-identity comparisons.
"""

import pygame
import sys
import os

from renderer import Renderer, WINDOW_W, WINDOW_H
from ui import GameState, UIController

ANIMATION_DURATION = 0.18  # seconds


class MoveAnimation:
    """Tracks an in-flight slide animation for the most recently applied move."""

    def __init__(self):
        self.active = False
        self.piece_value = None
        self.start_pos = None
        self.end_pos = None
        self.start_time = None
        self.to_square = None

    def start(self, move, renderer, board_after_move):
        (r0, c0) = move["path"][0]
        (r1, c1) = move["path"][-1]
        # board_after_move already has the piece at (r1, c1); that's the
        # value we animate (handles promotion: shows the promoted king
        # sliding in on its final hop, matching the original rules where
        # promotion applies at the end of the full move).
        self.piece_value = board_after_move[r1][c1]
        self.start_pos = renderer.square_center(r0, c0)
        self.end_pos = renderer.square_center(r1, c1)
        self.to_square = (r1, c1)
        self.start_time = pygame.time.get_ticks() / 1000.0
        self.active = True

    def current_pos(self):
        elapsed = pygame.time.get_ticks() / 1000.0 - self.start_time
        t = min(1.0, elapsed / ANIMATION_DURATION)
        t = 1 - (1 - t) ** 2  # ease-out
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * t
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * t
        if elapsed >= ANIMATION_DURATION:
            self.active = False
        return x, y


def main():
    pygame.init()
    pygame.display.set_caption("Checkers")

    icon_path = os.path.join("assets", "icon.png")
    if os.path.exists(icon_path):
        pygame.display.set_icon(pygame.image.load(icon_path))

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    renderer = Renderer(screen)
    state = GameState()
    controller = UIController(screen, state, renderer)

    anim = MoveAnimation()
    last_ply_count = state.ply_count

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                controller.handle_event(event)

        # A move just landed (human click or AI) if ply_count advanced.
        # A move just landed (human click or AI) if ply_count advanced.
        if state.ply_count != last_ply_count:
            if state.last_move is not None:
                anim.start(state.last_move, renderer, state.board)
            last_ply_count = state.ply_count

        controller.update()

        if state.ply_count != last_ply_count:
            if state.last_move is not None:
                anim.start(state.last_move, renderer, state.board)
            last_ply_count = state.ply_count

        renderer.draw_background()
        renderer.draw_board_frame()

        dest_squares = list(state.dest_map.keys())
        renderer.draw_highlights(state.selected, dest_squares)
        renderer.draw_last_move(state.last_move)

        if anim.active:
            board_for_draw = [row[:] for row in state.board]
            fr, fc = anim.to_square
            board_for_draw[fr][fc] = 0  # hide resting piece while it's sliding in
            renderer.draw_pieces(board_for_draw, animating_piece={
                "value": anim.piece_value,
                "pos": anim.current_pos(),
            })
        else:
            renderer.draw_pieces(state.board)

        controller.draw_sidebar()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
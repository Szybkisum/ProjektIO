import pygame
import sys
import os
from MinesweeperGame import MinesweeperGame
from DiffcultySettings import DIFFICULTY_LEVELS

TILE_SIZE = 30
MARGIN = 2
FONT_SIZE = 40
INFO_FONT_SIZE = 30

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (198, 198, 198)
DARK_GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 128)
RED = (255, 0, 0)
DARK_RED = (128, 0, 0)
GREEN = (0, 119, 0)
CYAN = (0, 255, 255)

NUMBER_COLORS = {
    '1': BLUE,
    '2': GREEN,
    '3': RED,
    '4': DARK_BLUE,
    '5': DARK_RED,
    '6': CYAN,
    '7': BLACK,
    '8': DARK_GRAY
}

class PygameMinesweeperGUI:
    def __init__(self, game, interactive = True):
        pygame.init()
        self.game = game
        self.screen_width = game.width * (TILE_SIZE + MARGIN) + MARGIN
        self.screen_height = game.height * (TILE_SIZE + MARGIN) + MARGIN + INFO_FONT_SIZE + 20
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Saper")
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.info_font = pygame.font.Font(None, INFO_FONT_SIZE)
        self.interactive = interactive

    def _draw_board(self):
        for y in range(self.game.height):
            for x in range(self.game.width):

                tile = self.game.board[y][x]
                left = x * (TILE_SIZE + MARGIN) + MARGIN
                top = y * (TILE_SIZE + MARGIN) + MARGIN + INFO_FONT_SIZE + 20
                rect = pygame.Rect(left, top, TILE_SIZE, TILE_SIZE)

                pygame.draw.rect(self.screen, GRAY, rect)
                if tile.is_revealed():
                    if tile.is_mine():
                        if (y, x) == self.game.loosing_tile:
                            pygame.draw.rect(self.screen, RED, rect)
                        pygame.draw.circle(self.screen, BLACK, rect.center, TILE_SIZE // 3)
                    elif not tile.is_empty():
                        pygame.draw.rect(self.screen, GRAY, rect)
                        num = tile.get_value()
                        color = NUMBER_COLORS.get(num, BLACK)
                        text_surface = self.font.render(num, True, color)
                        text_rect = text_surface.get_rect(center=rect.center)
                        self.screen.blit(text_surface, text_rect)
                else:
                    pygame.draw.rect(self.screen, WHITE, rect, 2)
                    if tile.is_flagged():
                        text_surface = self.font.render("!", True, BLACK)
                        text_rect = text_surface.get_rect(center=rect.center)
                        self.screen.blit(text_surface, text_rect)

    def _draw_game_state_message(self):
        message = ""
        color = BLACK
        if self.game.game_state == 'won':
            message = "WYGRANA!"
            color = GREEN
        elif self.game.game_state == 'lost':
            message = "PRZEGRANA!"
            color = RED
        if message:
            text_surface = self.info_font.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.screen_width // 2, INFO_FONT_SIZE // 2 + 10))
            self.screen.blit(text_surface, text_rect)

    def process_events(self):
        """Przetwarza zdarzenia, np. zamknięcie okna. Zwraca False, jeśli należy zakończyć."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN and self.interactive:
                    self._handle_mouse_click(event.pos)
        return True
            

    def draw(self):
        """Rysuje całą scenę gry."""
        self.screen.fill(DARK_GRAY)
        self._draw_board()
        self._draw_game_state_message()
        pygame.display.flip()

    def _handle_mouse_click(self, pos):
        click_y_offset = INFO_FONT_SIZE + 20
        adjusted_y = pos[1] - click_y_offset

        if adjusted_y < 0:
            return

        x = pos[0] // (TILE_SIZE + MARGIN)
        y = adjusted_y // (TILE_SIZE + MARGIN)

        if self.game.is_clickable(y, x):
            mouse_buttons = pygame.mouse.get_pressed()
            if mouse_buttons[0]:
                self.game.reveal_tile(y, x)
            elif mouse_buttons[2]:
                self.game.toggle_flag(y, x)      

    def quit(self):
        pygame.quit()
    
if __name__ == "__main__":
    DIFFICULTY = 'EASY'
    game = MinesweeperGame(**DIFFICULTY_LEVELS[DIFFICULTY])
    gui = PygameMinesweeperGUI(game)
    running = True
    while running:
        running = gui.process_events()
        gui.draw()
    gui.quit()
    sys.exit()
        
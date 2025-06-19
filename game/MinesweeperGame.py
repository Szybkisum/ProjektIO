import random
from Tile import Tile

class MinesweeperGame():
    def __init__(self, width, height, num_mines):
        self.width = width
        self.height = height
        self.num_mines = num_mines

        self.board = [[Tile() for _ in range(self.width)] for _ in range(self.height)]
        self.game_state = 'ongoing'

        self._place_mines()
        self._calculate_numbers()

    def _is_on_board(self, y, x):
        return 0 <= y < self.height and 0 <= x < self.width

    def _is_clickable(self, y, x):
        return self._is_on_board(y, x) and self.board[y][x].is_hidden_or_flagged() and self.game_state == 'ongoing'
    
    def _place_mines(self):
        mines_placed = 0
        while mines_placed < self.num_mines:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if not self.board[y][x].is_mine(): # '*' oznacza minę
                self.board[y][x].place_mine()
                mines_placed += 1

    def _calculate_numbers(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x].is_mine():
                    continue
                mine_count = self._count_mines_around(y, x)
                if mine_count > 0:
                    self.board[y][x].place_number(mine_count)

    def _count_mines_around(self, y, x):
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                check_y, check_x = y + i, x + j
                if self._is_on_board(check_y, check_x):
                    if self.board[check_y][check_x].is_mine():
                        count += 1
        return count

    def _check_win_condition(self):
        hidden_tiles = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x].is_hidden_or_flagged():
                    hidden_tiles += 1
        
        if hidden_tiles == self.num_mines:
            self.game_state = 'won'

    def _reveal_around(self, y, x):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                self.reveal_tile(y + i, x + j)


    def reveal_tile(self, y, x):
        if not self._is_clickable(y, x):
            return

        tile = self.board[y][x]
        tile.reveal()

        if tile.is_mine():
            self.game_state = 'lost'
            return

        if tile.is_empty():
            self._reveal_around(y, x)
    
        self._check_win_condition()

    def toggle_flag(self, y, x):
        if not self._is_clickable(y, x):
            return
        self.board[y][x].toggle_flag()


    def get_player_board_view(self):
        display_board = []
        for y in range(self.height):
            row_str = []
            for x in range(self.width):
                tile = self.board[y][x]
                if tile.is_flagged():
                    row_str.append('!')
                elif tile.is_hidden():
                    row_str.append('■')
                else:
                    row_str.append(tile.get_value())
            display_board.append(" ".join(row_str))
        return "\n".join(display_board)



if __name__ == '__main__':
    game = MinesweeperGame(width=9, height=9, num_mines=10)
    print("--- Plansza na starcie ---")
    print(game.get_player_board_view())

    game.reveal_tile(4, 4)
    game.toggle_flag(0, 1)
    
    print("\n--- Plansza po kilku ruchach ---")
    print(game.get_player_board_view())
    print(f"Stan gry: {game.game_state}")
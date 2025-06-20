import random
from Tile import Tile

class MinesweeperGame():
    def __init__(self, width, height, num_mines):
        self.width = width
        self.height = height
        self.num_mines = num_mines

        self.board = [[Tile() for _ in range(self.width)] for _ in range(self.height)]
        self.game_state = 'initial'
        self.loosing_tile = None
    def _is_on_board(self, y, x):
        return 0 <= y < self.height and 0 <= x < self.width

    def is_clickable(self, y, x):
        return self._is_on_board(y, x) and not self.board[y][x].is_revealed() and self.game_state in ('ongoing', 'initial')
    
    def _initialize(self, empty_y, empty_x):
        self._place_mines(empty_y, empty_x)
        self._calculate_numbers()
        self.game_state = 'ongoing'

    def _place_mines(self, empty_y, empty_x):
        mines_placed = 0
        taken = list(self._get_valid_neighbors(empty_y, empty_x))
        taken.append((empty_y, empty_x))
        while mines_placed < self.num_mines:
            y = random.randint(0, self.height - 1)
            x = random.randint(0, self.width - 1)
            curr = (y, x)
            if curr not in taken:
                self.board[y][x].place_mine()
                taken.append((y, x))
                mines_placed += 1
            
            

    def _get_valid_neighbors(self, y, x):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                
                check_y, check_x = y + i, x + j
                if self._is_on_board(check_y, check_x):
                    yield (check_y, check_x)

    def _get_all_tiles(self):
        for y in range(self.height):
            for x in range(self.width):
                yield (y, x, self.board[y][x])

    def _count_mines_around(self, y, x):
        return sum(1 for ny, nx in self._get_valid_neighbors(y, x) if self.board[ny][nx].is_mine())

    def _calculate_numbers(self):
        for y, x, tile in self._get_all_tiles():
            if tile.is_mine():
                continue
            mine_count = self._count_mines_around(y, x)
            if mine_count > 0:
                tile.place_number(mine_count)

    def _check_win_condition(self):        
        for y, x, tile in self._get_all_tiles():
            if not tile.is_mine() and tile.is_hidden():
                return
        if self.game_state == 'ongoing':
            self.game_state = 'won'

    def _reveal_around(self, y, x):
        for neighbor_y, neighbor_x in self._get_valid_neighbors(y, x):
            self.reveal_tile(neighbor_y, neighbor_x)

    def _reveal_all(self):
        for _, _, tile in self._get_all_tiles():
            tile.reveal()

    def reveal_tile(self, y, x):
        if not self.is_clickable(y, x):
            return

        if self.game_state == 'initial':
            self._initialize(y, x)

        tile = self.board[y][x]
        tile.reveal()

        if tile.is_mine():
            self.loosing_tile = (y, x)
            self.game_state = 'lost'
            self._reveal_all()
            return

        if tile.is_empty():
            self._reveal_around(y, x)
    
        self._check_win_condition()

    def toggle_flag(self, y, x):
        if not self.is_clickable(y, x):
            return
        self.board[y][x].toggle_flag()

    def get_observation(self):
        """
        Zwraca reprezentację planszy widoczną dla agenta (tablica 2D).
        """
        observation = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        for y, x, tile in self._get_all_tiles():
            if tile.is_flagged():
                observation[y][x] = 'F'
            elif tile.is_hidden():
                observation[y][x] = 'H'
            else:
                observation[y][x] = tile.get_value()
        return observation

    def get_player_board_view(self):
        display_board = []
        for y in range(self.height):
            row_str = []
            for x in range(self.width):
                tile = self.board[y][x]
                if tile.is_flagged():
                    row_str.append('!')
                elif tile.is_hidden():
                    row_str.append('#')
                else:
                    row_str.append(tile.get_value())
            display_board.append(" ".join(row_str))
        return "\n".join(display_board)


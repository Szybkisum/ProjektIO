from MinesweeperGame import MinesweeperGame

class SimpleAgent:

    def _get_neighbors_coords(self, y, x, height, width):
        """Prosta metoda pomocnicza do generowania koordynatów sąsiadów."""
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                check_y, check_x = y + i, x + j
                if 0 <= check_y < height and 0 <= check_x < width:
                    yield (check_y, check_x)

    def make_move(self, observation):
        """
        Otrzymuje obserwację (tablicę 2D) i zwraca akcję do wykonania.
        Akcja to krotka, np. ('reveal', y, x) lub ('flag', y, x) lub None.
        """
        height = len(observation)
        width = len(observation[0])

        for y in range(height):
            for x in range(width):
                tile_value = observation[y][x]
                if tile_value.isdigit():
                    n = int(tile_value)
                    neighbors_coords = self._get_neighbors_coords(y, x, height, width)
                    
                    hidden_neighbors = []
                    flagged_count = 0
                    
                    for ny, nx in neighbors_coords:
                        if observation[ny][nx] == 'H':
                            hidden_neighbors.append((ny, nx))
                        elif observation[ny][nx] == 'F':
                            flagged_count += 1
                    
                    if len(hidden_neighbors) > 0 and (len(hidden_neighbors) + flagged_count) == n:
                        ny, nx = hidden_neighbors[0]
                        print(f"Agent: Znalazłem minę na ({nx}, {ny}) na podstawie pola ({x}, {y})")
                        return ('flag', ny, nx)

                    if len(hidden_neighbors) > 0 and flagged_count == n:
                        ny, nx = hidden_neighbors[0]
                        print(f"Agent: Znalazłem bezpieczne pole na ({nx}, {ny}) na podstawie pola ({x}, {y})")
                        return ('reveal', ny, nx)

        return None


if __name__ == '__main__':

    game = MinesweeperGame(width=9, height=9, num_mines=10)
    agent = SimpleAgent()

    game.reveal_tile(4, 4)
    print("--- Plansza na starcie ---")
    print(game.get_player_board_view())
    print("-" * 20)

    while game.game_state == 'ongoing':
        current_observation = game.get_observation()
        
        action = agent.make_move(current_observation)

        if action is None:
            print("Agent nie znalazł pewnego ruchu.")
            break

        action_type, y, x = action
        if action_type == 'reveal':
            game.reveal_tile(y, x)
        elif action_type == 'flag':
            game.toggle_flag(y, x)
        print(game.get_player_board_view())
        print("-" * 20)

    print(f"\nGra zakończona! Stan: {game.game_state}")
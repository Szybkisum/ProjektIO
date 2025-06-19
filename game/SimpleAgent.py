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
                        return ('flag', ny, nx)

                    if len(hidden_neighbors) > 0 and flagged_count == n:
                        ny, nx = hidden_neighbors[0]
                        return ('reveal', ny, nx)

        return None
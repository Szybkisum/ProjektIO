import random
from SimpleAgent import SimpleAgent
from MinesweeperGame import MinesweeperGame


class ProbabilisticAgent():

    def __init__(self):
        self.simple_agent = SimpleAgent()

    def make_move(self, observation):
        certain_move = self.simple_agent.make_move(observation)

        if certain_move is not None:
            return certain_move
        else:  
            hidden_tiles = []
            for y in range(len(observation)):
                for x in range(len(observation[0])):
                    if observation[y][x] == 'H':
                        hidden_tiles.append((y, x))
            
            if hidden_tiles:
                random_y, random_x = random.choice(hidden_tiles)
                print(f"Agent (Zaawansowany): Brak pewnych ruchów. Zgaduję pole ({random_x}, {random_y})")
                return ('reveal', random_y, random_x)

        return None

if __name__ == '__main__':

    game = MinesweeperGame(width=9, height=9, num_mines=10)
    agent = ProbabilisticAgent()

    print("--- Plansza na starcie ---")
    print(game.get_player_board_view())
    print("-" * 20)

    while game.game_state == 'ongoing':
        current_observation = game.get_observation()
        
        action = agent.make_move(current_observation)

        action_type, y, x = action
        if action_type == 'reveal':
            game.reveal_tile(y, x)
        elif action_type == 'flag':
            game.toggle_flag(y, x)
        print(game.get_player_board_view())
        print("-" * 20)

    print(f"\nGra zakończona! Stan: {game.game_state}")
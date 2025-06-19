from MinesweeperGame import MinesweeperGame
from SimpleAgent import SimpleAgent
from GuessingAgent import GuessingAgent
from DiffcultySettings import DIFFICULTY_LEVELS
import random

def evaluate_agent(agent_class, difficulty_settings, num_games=100):
    """
    Uruchamia symulację N gier dla danego agenta i zwraca statystyki.
    """
    stats = {'won': 0, 'lost': 0}
    agent = agent_class()

    for i in range(num_games):

        game = MinesweeperGame(**difficulty_settings)
        
        start_y, start_x = random.randint(0, game.height - 1), random.randint(0, game.width - 1)
        game.reveal_tile(start_y, start_x)

        while game.game_state == 'ongoing':
            observation = game.get_observation()
            action = agent.make_move(observation)

            if action is None:
                stats['lost'] += 1
                break
            
            action_type, y, x = action
            if action_type == 'reveal':
                game.reveal_tile(y, x)
            elif action_type == 'flag':
                game.toggle_flag(y, x)
        
        if game.game_state == 'won':
            stats['won'] += 1
        elif game.game_state == 'lost':
            stats['lost'] += 1

    return stats

def print_stats(agent_name, stats, num_games):
    print(f"\n--- Wyniki dla: {agent_name} ({num_games} gier) ---")
    win_rate = (stats['won'] / num_games) * 100
    lost_rate = (stats['lost'] / num_games) * 100
    
    print(f"Wygrane: {stats['won']} ({win_rate:.2f}%)")
    print(f"Przegrane: {stats['lost']} ({lost_rate:.2f}%)")


if __name__ == '__main__':
    NUM_GAMES = 100
    DIFFICULTY = "EASY"

    simple_stats = evaluate_agent(SimpleAgent, DIFFICULTY_LEVELS[DIFFICULTY], NUM_GAMES)
    print_stats("SimpleAgent", simple_stats, NUM_GAMES)
    
    advanced_stats = evaluate_agent(GuessingAgent, DIFFICULTY_LEVELS[DIFFICULTY], NUM_GAMES)
    print_stats("GuessingAgent (z losowym ruchem)", advanced_stats, NUM_GAMES)
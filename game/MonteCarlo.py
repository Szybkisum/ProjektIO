import numpy as np
from MinesweeperGame import MinesweeperGame
from SimpleAgent import SimpleAgent
from GuessingAgent import GuessingAgent
from dqn_agent import DQNAgent
from dqn_architecture import preprocess_observation
from DiffcultySettings import DIFFICULTY_LEVELS
import os

def run_evaluation(agent, difficulty_settings, num_games=100, agent_type='SIMPLE'):
    """Uniwersalna funkcja do ewaluacji dowolnego agenta."""
    stats = {'won': 0, 'lost': 0}

    print(f"Rozpoczynam ewaluację agenta: {agent_type} ({num_games} gier)...")
    
    for i in range(num_games):
        game = MinesweeperGame(**difficulty_settings)
        start_y, start_x = np.random.randint(0, game.height), np.random.randint(0, game.width)
        game.reveal_tile(start_y, start_x)

        done = False
        while not done and game.game_state == 'ongoing':
            observation = game.get_observation()
            if agent_type == "DQN":
                state = preprocess_observation(observation)
                action_idx = agent.act(state, observation)
                y, x = divmod(action_idx, difficulty_settings['width'])
                action = ('reveal', y, x)
            else:
                action = agent.make_move(observation)
            
            if action is None:
                break
            
            action_type, y, x = action
            if action_type == 'reveal':
                game.reveal_tile(y, x)
            elif action_type == 'flag':
                game.toggle_flag(y, x)
            
            done = game.game_state != 'ongoing'

        if game.game_state == 'won':
            stats['won'] += 1
        else:
            stats['lost'] += 1
            
    return stats

def print_final_results(results, num_games):
    """Drukuje finalną tabelę porównawczą."""
    print("\n\n" + "="*60)
    print(" " * 15 + "FINALNE WYNIKI PORÓWNAWCZE")
    print("="*60)
    
    for agent_name, result in results.items():
        difficulty = result['difficulty']
        stats = result['stats']
        
        print(f"\n--- Wyniki dla: {agent_name} na poziomie {difficulty} ({num_games} gier) ---")
        if stats:
            win_rate = (stats['won'] / num_games) * 100
            print(f"    SKUTECZNOŚĆ WYGRANYCH: {win_rate:.2f}% ({stats['won']}/{num_games})")
        else:
            print("    Brak wyników (prawdopodobnie błąd wczytania modelu).")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    NUM_GAMES = 1000
    DIFFICULTY_NAME = "BABY"
    DIFFICULTY_SETTINGS = DIFFICULTY_LEVELS[DIFFICULTY_NAME]

    all_results = {}

    # --- Ewaluacja Agenta Prostego ---
    simple_agent = SimpleAgent()
    simple_stats = run_evaluation(simple_agent, DIFFICULTY_SETTINGS, NUM_GAMES, agent_type='SIMPLE')
    all_results['SimpleAgent'] = {'stats': simple_stats, 'difficulty': DIFFICULTY_NAME}
    
    # --- Ewaluacja Agenta Zgadującego ---
    guessing_agent = GuessingAgent()
    guessing_stats = run_evaluation(guessing_agent, DIFFICULTY_SETTINGS, NUM_GAMES, agent_type='GUESSING')
    all_results['GuessingAgent'] = {'stats': guessing_stats, 'difficulty': DIFFICULTY_NAME}

    # --- Ewaluacja Agenta DQN ---
    MODEL_PATH = f"./dqn_model_{DIFFICULTY_NAME}.keras"
    if os.path.exists(MODEL_PATH):
        dqn_agent = DQNAgent(
            state_shape=(DIFFICULTY_SETTINGS['height'], DIFFICULTY_SETTINGS['width'], 12),
            num_actions=DIFFICULTY_SETTINGS['height'] * DIFFICULTY_SETTINGS['width']
        )
        dqn_agent.load(MODEL_PATH)
        dqn_agent.epsilon = 0.0
        dqn_stats = run_evaluation(dqn_agent, DIFFICULTY_SETTINGS, NUM_GAMES, agent_type='DQN')
        all_results['DQNAgent'] = {'stats': dqn_stats, 'difficulty': DIFFICULTY_NAME}
    else:
        print(f"\nOSTRZEŻENIE: Nie znaleziono pliku modelu '{MODEL_PATH}'. Pomijam ewaluację agenta DQN.")

    print_final_results(all_results, NUM_GAMES)
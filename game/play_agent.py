import pygame
import random
from MinesweeperGame import MinesweeperGame
from MinesweeperGUI import PygameMinesweeperGUI
from dqn_agent import DQNAgent
from dqn_architecture import preprocess_observation
from DiffcultySettings import DIFFICULTY_LEVELS
from SimpleAgent import SimpleAgent
from GuessingAgent import GuessingAgent

# Opcje: "SIMPLE", "GUESSING", "DQN"
AGENT_TYPE = "DQN"
DIFFICULTY_NAME = "BABY"
MODEL_PATH = f"./dqn_model_{DIFFICULTY_NAME}_10500.keras" 
DELAY_BETWEEN_MOVES = 500

def initialize_agent(agent_type, difficulty_settings, model_path):
    """Tworzy i konfiguruje wybranego agenta."""
    if agent_type == "SIMPLE":
        return SimpleAgent()
    if agent_type == "GUESSING":
        return GuessingAgent()
    if agent_type == "DQN":
        height, width = difficulty_settings['height'], difficulty_settings['width']
        agent = DQNAgent((height, width, 12), height * width)
        try:
            agent.load(model_path)
            agent.epsilon = 0.0
            return agent
        except Exception as e:
            print(f"BŁĄD: Nie udało się wczytać modelu z '{model_path}'.")
            return None  
    return None

def run_demonstration():
    settings = DIFFICULTY_LEVELS[DIFFICULTY_NAME]
    
    agent = initialize_agent(AGENT_TYPE, settings, MODEL_PATH)
    if agent is None:
        print("Nie udało się zainicjalizować agenta. Zamykanie.")
        return

    game = MinesweeperGame(**settings)
    gui = PygameMinesweeperGUI(game, False)

    running = True
    last_move_time = pygame.time.get_ticks()
    game_over_pause_time = None

    while running:

        if not gui.process_events():
            running = False
            continue
        
        current_time = pygame.time.get_ticks()

        if game.game_state in ('initial', 'ongoing') and current_time - last_move_time > DELAY_BETWEEN_MOVES:
                
            last_move_time = current_time
            observation = game.get_observation()

            if isinstance(agent, DQNAgent):
                state = preprocess_observation(observation)
                action_idx = agent.act(state, observation)
                y, x = divmod(action_idx, settings['width'])
                action = ('reveal', y, x)
            else:
                action = agent.make_move(observation)

            if action is None:
                if game.game_state == 'initial':
                    start_y, start_x = random.randint(0, game.height - 1), random.randint(0, game.width - 1)
                    action = ('reveal', start_y, start_x)
                else:
                    game.game_state = 'lost'
            if action:
                action_type, y, x = action
                if action_type == 'reveal':
                    game.reveal_tile(y, x)
                elif action_type == 'flag':
                    game.toggle_flag(y, x)

        elif game.game_state in ('won', 'lost'):
            if game_over_pause_time is None:
                game_over_pause_time = current_time
            
            if current_time - game_over_pause_time > 2000:
                game = MinesweeperGame(**settings)
                gui.game = game
                game_over_pause_time = None         
        gui.draw()
    gui.quit()

if __name__ == '__main__':
    run_demonstration()
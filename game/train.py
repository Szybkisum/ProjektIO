from MinesweeperGame import MinesweeperGame
from dqn_agent import DQNAgent
from dqn_architecture import preprocess_observation
from DiffcultySettings import DIFFICULTY_LEVELS
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def get_reward(game_state_after, is_illegal_move):
    """Definiuje system nagród dla agenta."""
    if is_illegal_move:
        return -5.0
    if game_state_after == 'won':
        return 100.0
    if game_state_after == 'lost':
        return -100.0
    return 1.0

def plot_history(history, name):
    """Tworzy i zapisuje wykresy postępu uczenia."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Postęp Treningu Agenta DQN')

    axs[0].plot(history['rewards'], color='green')
    axs[0].set_title('Suma nagród w epizodzie')
    axs[0].set_ylabel('Suma Nagród')
    
    moving_avg_rewards = np.convolve(history['rewards'], np.ones(100)/100, mode='valid')
    axs[0].plot(moving_avg_rewards, color='darkgreen', label='Średnia krocząca (100 epizodów)')
    axs[0].legend()

    axs[1].plot(history['avg_loss'], color='red')
    axs[1].set_title('Średnia strata (Loss) w epizodzie')
    axs[1].set_ylabel('Średnia Strata')

    axs[2].plot(history['epsilon'], color='blue')
    axs[2].set_title('Wygaszanie Epsilon')
    axs[2].set_xlabel('Epizod')
    axs[2].set_ylabel('Wartość Epsilon')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(name)
    print(f"\nZapisano wykresy postępu do pliku {name}")


def train_dqn():
    EPISODES = 10000
    MAX_STEPS = 1000
    DIFFICULTY = "BABY"
    settings = DIFFICULTY_LEVELS[DIFFICULTY]
    height, width, num_mines = settings['height'], settings['width'], settings['num_mines']

    num_actions = height * width

    state_shape = (height, width, 12)
    agent = DQNAgent(state_shape, num_actions)
    
    history = {'rewards': [], 'avg_loss': [], 'epsilon': []}

    for e in range(EPISODES):
        game = MinesweeperGame(**settings)
        
        start_y, start_x = np.random.randint(0, height), np.random.randint(0, width)
        game.reveal_tile(start_y, start_x)
        
        state = preprocess_observation(game.get_observation())
        
        done = False
        total_reward = 0
        episode_losses = []
        step = 0
        
        while not done:
            step += 1
            if step > MAX_STEPS:
                break

            action_idx = agent.act(state, game.get_observation())
            y, x = divmod(action_idx, width)
            
            is_illegal = game.get_observation()[y][x] != 'H'
            if not is_illegal:
                game.reveal_tile(y, x)
            
            reward = get_reward(game.game_state, is_illegal)
            total_reward += reward

            done = game.game_state != 'ongoing'
            next_state = preprocess_observation(game.get_observation())
            
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            
            if step % 4 == 0 or len(agent.replay_buffer) == agent.replay_buffer.buffer.maxlen:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)

        history['rewards'].append(total_reward)
        history['avg_loss'].append(np.mean(episode_losses) if episode_losses else 0)
        history['epsilon'].append(agent.epsilon)
        
        agent.update_target_model()
        
        print(f"Epizod: {e+1}/{EPISODES}, Nagroda: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Stan gry: {game.game_state}")
        
        if (e + 1) % 500 == 0:
            save_path = f"./minesweeper-dqn-{DIFFICULTY}"
            agent.save(f"{save_path}.weights.h5")
            print(f"Zapisano model do pliku: {save_path}")
            plot_history(history, f"{save_path}-episode-{e + 1}.png")

    plot_history(history, f"final_training_progress_{DIFFICULTY}.png")


if __name__ == '__main__':
    train_dqn()
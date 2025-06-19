import numpy as np # type: ignore
import random
from collections import deque
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def create_dqn_model(height, width, num_actions):
    """
    Tworzy model konwolucyjnej sieci neuronowej (CNN) dla naszego agenta.
    """
    model = Sequential()
    model.add(Input((height, width, 12)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
    return model

class ReplayBuffer:
    """
    Bufor do przechowywania i losowania doświadczeń agenta.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Dodaje doświadczenie do bufora."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Losuje paczkę doświadczeń z bufora."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def preprocess_observation(observation):
    """
    Konwertuje obserwację (lista list stringów) na tensor numeryczny 
    dla sieci neuronowej (one-hot encoding).
    """
    height = len(observation)
    width = len(observation[0])
    # 12 kanałów: 0-8 dla cyfr, 9-puste, 10-ukryte, 11-oflagowane
    state = np.zeros((height, width, 12), dtype=np.float32)

    for r in range(height):
        for c in range(width):
            cell = observation[r][c]
            if cell.isdigit():
                state[r, c, int(cell)] = 1.0
            elif cell == ' ':
                state[r, c, 9] = 1.0
            elif cell == 'H':
                state[r, c, 10] = 1.0
            elif cell == 'F':
                state[r, c, 11] = 1.0
    return state
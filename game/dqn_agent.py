import numpy as np # type: ignore
import random
from dqn_architecture import create_dqn_model, ReplayBuffer

class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        
        # Hiperparametry
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.batch_size = 256

        self.model = create_dqn_model(state_shape[0], state_shape[1], num_actions)
        self.target_model = create_dqn_model(state_shape[0], state_shape[1], num_actions)
        self.update_target_model()

    def update_target_model(self):
        """Kopiuje wagi z sieci głównej do sieci docelowej."""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, observation):

        if np.random.rand() <= self.epsilon:
            legal_actions = [i for i, tile in enumerate(np.array(observation).flatten()) if tile == 'H']
            if not legal_actions:
                return random.randrange(self.num_actions)
            return random.choice(legal_actions)
        
        state_tensor = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state_tensor, verbose=0)[0]
        
        for i in range(self.num_actions):
            y, x = divmod(i, self.state_shape[1])
            if observation[y][x] != 'H':
                q_values[i] = -np.inf
        
        action_idx = np.argmax(q_values)

        if q_values[action_idx] == -np.inf:
            legal_actions = [i for i, tile in enumerate(np.array(observation).flatten()) if tile == 'H']
            if not legal_actions:
                print("NAJN")
                return random.randrange(self.num_actions)
            return random.choice(legal_actions)

        return action_idx

    def remember(self, state, action, reward, next_state, done):
        """Zapisuje doświadczenie w buforze powtórek."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def replay(self):
        """Uczy sieć na podstawie losowej paczki z bufora, używając logiki DDQN."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        minibatch = self.replay_buffer.sample(self.batch_size)
        
        states = np.array([transition[0] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        
        current_q_values_main_model = self.model.predict(states, verbose=0)
        future_q_values_main_model = self.model.predict(next_states, verbose=0)
        future_q_values_target_model = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                best_action_in_next_state = np.argmax(future_q_values_main_model[i])
                q_value_from_target_model = future_q_values_target_model[i][best_action_in_next_state]
                new_q = reward + self.gamma * q_value_from_target_model
            else:
                new_q = reward
            
            current_q_values_main_model[i][action] = new_q
        
        history = self.model.fit(states, current_q_values_main_model, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
            
    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)
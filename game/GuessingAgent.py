import random
from SimpleAgent import SimpleAgent

class GuessingAgent():

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
                return ('reveal', random_y, random_x)

        return None
from dqn_architecture import create_dqn_model
from tensorflow.keras.utils import plot_model # type: ignore
from DiffcultySettings import DIFFICULTY_LEVELS

DIFFICULTY_NAME = "BABY"
settings = DIFFICULTY_LEVELS[DIFFICULTY_NAME]

height = settings['height']
width = settings['width']
num_actions = height * width

model = create_dqn_model(height, width, num_actions)

output_filename = f'dqn_model_architecture_{DIFFICULTY_NAME}.png'
plot_model(
    model,
    to_file=output_filename,
    show_shapes=True,
    show_layer_names=True,
    show_layer_activations=True
)
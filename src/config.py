import numpy as np


###
# STANDARD PARAMETERS
###

FIGURES_PATH = "/Users/kisslucasara/Documents/levi-research-project/figures/"

# demand model parameters
DEMAND_MODEL_PRICE_GRANULARITY = 0.001

# MDP main parameters
K = 200  # number of episodes used for evaluation
NUM_TIMESTEPS = 60  # number of time steps per episode
NUM_PRODUCTS = 2

# Environment parameters
P_MIN = np.array([5.0, 5.0])
P_DIFF = np.array([0.01, 0.01])
P_MAX = np.array([9.0, 9.0])
N_SNAP_DAYS = 10
CALENDAR_SEED = 42
NOISE_RANGE = 1
ENV_SEED = 42
# num. of episodes is added to have no overlap between seeds
SEED_LIST = [1024, 1024 + K, 1024 + 2 * K]

# PPO hyperparameters
PPO_HYPERPARAMS = {
    "learning_rate": 0.0001,
    "batch_size": 64,  # default
    "clip_range": 0.2,  # default
    "seed": 42,
}
PPO_TRAINING_EPISODES = 15000

###
# EXPERIMENT 1 PARAMETERS
###

RANDOM_PROBS_LIST = [i * (1 / 9) for i in range(10)]

###
# EXPERIMENT 2 PARAMETERS
###

NUM_INITIAL_RUNS = 100
FAIRNESS_WEIGHTS_LIST = [i * (1 / 9) for i in range(10)]

###
# EXPERIMENT 3 PARAMETERS
###

RANDOM_SENS_PROBS_LIST = [i * (1 / 9) for i in range(10)]

import numpy as np
import gymnasium as gym
from typing import Optional

class FMCGEnv(gym.Env):

    def __init__(self,
                 p_min: np.ndarray,
                 p_max: np.ndarray,
                 p_diff: np.ndarray,
                 alpha: list,
                 gamma: list,
                 beta: list,
                 budget: list,
                 real_budget: list,
                 noise_range: int,
                 H: int = 365,
                 sens_attr_mapping: Optional[list] = None
                 ):
        """ Initialize FMCG Gymnasium environment.

        Args:
            p_min (np.array): minimum price vector of (L)
            p_max (np.array): maximum price vector of (L)
            p_diff (np.array): price step vector of (L)
            alpha (List[np.ndarray]): demand core state param. (base budget share)
            gamma (List[np.ndarray]): demand core state param. ((cross-)price elasticities)
            beta (List[float]): demand core state param. (income elasticity)
            budget (List[float]): budget of representative customer for a time step
            real_budget (List[float]): real budget of representative customer for a time step
            noise_range (int): maximum value of noise (symmetric)
            H (int): length of episode (defaults to 365),
            sens_attr_mapping (list, optional): mapping of time steps to sensitive attributes
        """
        # input validation
        assert p_min.shape == p_max.shape == p_diff.shape, "Length of price vectors does not match."

        # main parameters
        self.p_min = p_min
        self.p_max = p_max
        self.p_diff = p_diff
        # number of products
        self.L = p_min.shape[0]
        self.H = H
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.x = budget
        self.real_budget = real_budget
        self.noise_range = noise_range

        # set sensitive attribute mapping
        if sens_attr_mapping is not None:
            if not (len(sens_attr_mapping) == self.H):
                raise ValueError(
                    "Sensitive attribute mapping with respect to t should be of length H."
                )
            else:
                self._t_to_sens_attr = sens_attr_mapping
        else:
            # initialize randomly a sensitive attribute
            self._t_to_sens_attr = np.random.randint(0, 2, size=self.H)

        # check user input
        if not self._is_p_consistent(self.p_min, self.p_max, self.p_diff):
            raise ValueError(
                "Inconsistent price parameters, (p_max - p_min) / p_diff is not an integer."
            )

        # Initialize state
        # Using -1 as "uninitialized" state
        self._demand = -1.0 * np.ones((self.L))
        self._t = -1
        self._sensitive_attr = -1

        # Define what the agent can observe
        # due to normalization, the maximum demand is 1.0
        self.observation_space = gym.spaces.Dict({
            "demand": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.L,),
                dtype=np.float32
            ),
            "t": gym.spaces.Discrete(self.H),
            "sensitive_attr": gym.spaces.Discrete(2)
        })
        
        # The available actions correspond to possible price points
        num_actions = self._get_num_actions() # round((self.p_max - self.p_min) / self.p_diff) + 1
        # list of L actions corresponding to prices
        self.action_space = gym.spaces.MultiDiscrete(num_actions)

        # maximum demand
        self._max_d = self._get_max_demand()
    
    def reset(self,
                seed: Optional[int] = None,
                options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # At the start of the episode t = 0
        self._t = 0
        self._sensitive_attr = self._t_to_sens_attr[self._t]

        # At time point 0 the demand is for a random price vector
        init_price = self._action_to_price(self.np_random.integers(low=0,
                                                      high=self.action_space.nvec,
                                                      size=self.L))
        self._demand = self._demand_model(init_price)

        # clip demand to sensible range
        self._demand = np.clip(self._demand, 0.0, self._max_d)

        observation = self._get_obs()
        info = self._get_info()
 
        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action to a price
        p = self._action_to_price(action)

        # Observe demand based on core state and noise
        self._demand = self._demand_model(p) + self.np_random.integers(low=-self.noise_range,
                                                      high=self.noise_range,
                                                      size=self.L)
        # clip demand to sensible range
        self._demand = np.clip(self._demand, 0.0, self._max_d)

        # Check if episode ended
        terminated = (self._t == (self.H - 1))

        # Update time step
        self._t += 1
        if not terminated:
            self._sensitive_attr = self._t_to_sens_attr[self._t]   

        # We don't use truncation in this environment
        truncated = False

        # Reward is the obtained revenue
        reward = p @ self._demand

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with demand and time point
        """
        # apply min-max scaling to demand
        demand_out = (self._demand / self._max_d).astype(np.float32)

        return {"demand":  demand_out,
                 "t": int(self._t),
                 "sensitive_attr": int(self._sensitive_attr)}

    def _get_info(self):
        """ Placeholder method for aiding debugging
        """
        return {}
    
    def _action_to_price(self, actions):
        """Map action numbers to actual price points.
        """ 
        # rounding guards against floating point inconsistencies
        prices = np.round(self.p_min + actions * self.p_diff, 10)

        return prices

    def _demand_model(self,prices):
        """ Returns demand vector from price vector from estimated AIDS model.
        """
        log_prices = np.log(prices)

        # from AIDS equation, first obtain budget shares:
        first_term = self.alpha[self._sensitive_attr]
        second_term = self.gamma[self._sensitive_attr] @ log_prices
        third_term = self.beta[self._sensitive_attr] * np.log(self.real_budget[self._sensitive_attr])
        budget_shares = first_term + second_term + third_term

        # the demand is the budget shares times the budget over the prices
        demand_vec = (budget_shares * self.x[self._sensitive_attr]) / prices

        return demand_vec

    def _get_max_demand(self):
        """ The maximum demand over all products.
        """
        # the maximum demand is when 100% of the budget is spent on the cheapest item
        min_price = np.min(self.p_min)
        max_d = self.x[self._sensitive_attr] / min_price
        return max_d

    @staticmethod
    def _is_p_consistent(p_min:float,
                         p_max:float,
                         p_diff:float) -> bool:
        """ Checks consistency of price parameters.
        
        Args:
            p_min (np.array): minimum price vector of (L)
            p_max (np.array): maximum price vector of (L)
            p_diff (np.array): price step vector of (L)

        Returns:
            bool: indicates whether price params. are consistent
        """
        # to have an int. number of steps, the price range
        # should be divisible by p_diff
        n = (p_max - p_min) / p_diff
        return np.allclose(n, np.round(n), rtol=1e-9)

    def _get_num_actions(self):
        """Number of discrete price points per product (inclusive of both endpoints).
        """
        n_steps = (self.p_max - self.p_min) / self.p_diff
        return (np.round(n_steps) + 1).astype(int)

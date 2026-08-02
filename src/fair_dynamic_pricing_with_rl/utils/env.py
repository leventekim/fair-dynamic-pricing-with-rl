import numpy as np
import gymnasium as gym
from typing import Optional


class FMCGEnv(gym.Env):
    def __init__(
        self,
        p_min: np.ndarray,
        p_max: np.ndarray,
        p_diff: np.ndarray,
        betas: list,
        noise_range: int,
        max_demand: int,
        n_snap_days: int,
        calendar_seed: Optional[int] = None,
        T: Optional[int] = 60,
    ):
        """Initialize FMCG Gymnasium environment.

        Args:
            p_min (np.array): minimum price vector of products (L)
            p_max (np.array): maximum price vector of products (L)
            p_diff (np.array): price step vector of products (L)
            betas (List[np.ndarray]): demand core state param. for non-SNAP and SNAP households
            noise_range (int): maximum value of noise (symmetric)
            max_demand (int): the maximum daily demand considering all L products
            n_snap_days (int): number of SNAP days in a 30-day month.
            calendar_seed (int, optional): seed used for constructing SNAP calendar
            T (int, optional): length of episode (defaults to 60),
        """
        # input validation
        assert p_min.shape == p_max.shape == p_diff.shape, (
            "Length of price vectors does not match."
        )

        # main parameters
        self.p_min = p_min
        self.p_max = p_max
        self.p_diff = p_diff
        # number of products
        self.L = p_min.shape[0]
        self.T = T
        self.n_snap_days = n_snap_days
        # demand model parameters
        self.betas = betas
        self.noise_range = noise_range
        # maximum demand
        self._max_demand = max_demand

        ###
        # CHECK USER INPUT
        ###
        if not self._is_p_consistent(self.p_min, self.p_max, self.p_diff):
            raise ValueError(
                "Inconsistent price parameters, (p_max - p_min) / p_diff is not an integer."
            )
        if not 0 <= self.n_snap_days <= 30:
            raise ValueError("Number of SNAP days must be in [0, 30]")

        ###
        # INITIALIZE STATE
        ###

        # Using -1 as "uninitialized" state
        self._demand = -1.0 * np.ones(self.L)
        self._t = -1
        self._sensitive_attr = -1

        # Define what the agent can observe
        # due to normalization, the maximum demand is 1.0
        self.observation_space = gym.spaces.Dict(
            {
                "demand": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.L,), dtype=np.float32
                ),
                "t": gym.spaces.Discrete(self.T),
                "sensitive_attr": gym.spaces.Discrete(2),
            }
        )

        # The available actions correspond to possible price points
        num_actions = (
            self._get_num_actions()
        )  # round((self.p_max - self.p_min) / self.p_diff) + 1
        # list of L actions corresponding to prices
        self.action_space = gym.spaces.MultiDiscrete(num_actions)

        ###
        # SET SENSITIVE ATTRIBUTE MAPPING
        ###
        rng = np.random.default_rng(calendar_seed)
        idx = rng.choice(30, size=self.n_snap_days, replace=False)
        self.mask = np.zeros(30, dtype=bool)
        self.mask[idx] = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration

        Returns:
            tuple: (observation, info) for the initial state
        """
        # seeding the random number generator
        super().reset(seed=seed)

        ### INITIALIZATION
        # At the start of the episode t = 0
        self._t = 0

        # get initial sensitive attribute
        self._sensitive_attr = self._t_to_sens_attr(self._t)

        # action history
        self.action_history = np.zeros((self.T, self.L))

        ### INITIAL DEMAND
        # At time point 0 the demand is for a random price vector
        init_price = self._action_to_price(
            self.np_random.integers(low=0, high=self.action_space.nvec, size=self.L)
        )
        # during initialization it is assumed that d_t = d_{t-1}
        # therefore: d_t = ... + beta d_{t}
        # that implies d_t = ... / (1-beta)
        self._demand = self._demand_model(init_price, np.zeros((self.L)))
        self._demand = self._demand / (1 - self.betas[self._sensitive_attr][-1, :])

        # clip demand to sensible range
        self._demand = np.clip(self._demand, 0.0, self._max_demand)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action to a price and store it in history
        p = self._action_to_price(action)
        self.action_history[self._t, :] = action

        # Observe demand based on core state and noise
        demand_t = self._demand_model(p, self._demand) + self.np_random.integers(
            low=-self.noise_range, high=self.noise_range + 1, size=self.L
        )
        # clip demand to sensible range
        self._demand = np.clip(demand_t, 0.0, self._max_demand)

        # Check if episode ended
        terminated = self._t == (self.T - 1)

        # Update time step
        if not terminated:
            self._t += 1
            self._sensitive_attr = self._t_to_sens_attr(self._t)

        # We don't use truncation in this environment
        truncated = False

        # Reward is the obtained revenue
        reward = self._get_reward(p, self._demand)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with demand and time point
        """
        # apply min-max scaling to demand
        demand_out = (self._demand / self._max_demand).astype(np.float32)

        return {
            "demand": demand_out,
            "t": int(self._t),
            "sensitive_attr": int(self._sensitive_attr),
        }

    def _get_info(self):
        """Placeholder method for aiding debugging"""
        return {}

    def _action_to_price(
        self,
        actions: np.ndarray,
    ):
        """Map action numbers to actual price points."""
        # rounding guards against floating point inconsistencies
        prices = np.round(self.p_min + actions * self.p_diff, 10)

        return prices

    def _demand_model(
        self,
        prices: np.ndarray,
        demand_t_1: np.ndarray,
    ):
        """Returns demand vector from price vector from estimated demand model.

        Args:
            prices (np.array): price vector of L products (L,)
            demand_t_1 (np.array): demand vector at timestep t-1 (L,)

        Returns:
            demand_vec (np.array): demand vector at time t (L,)
        """
        # create dataframe similar in structure to design matrix during training
        lp = np.log(prices)
        iu = np.triu_indices(self.L, k=1)

        design_row = np.concatenate(
            [
                [1.0],
                lp,
                (lp[:, None] * lp[None, :])[iu],
                lp**2,
                demand_t_1,
            ]
        )

        # using estimated coefficients from demand model obtain demand vector
        demand_vec = design_row @ self.betas[self._sensitive_attr]

        return demand_vec

    @staticmethod
    def _is_p_consistent(p_min: float, p_max: float, p_diff: float) -> bool:
        """Checks consistency of price parameters.

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
        """Number of discrete price points per product (inclusive of both endpoints)."""
        n_steps = (self.p_max - self.p_min) / self.p_diff
        return (np.round(n_steps) + 1).astype(int)

    def _t_to_sens_attr(
        self,
        t: int,
    ):
        """Returns the sensitive attribute at t
        Args:
            t (int): time step index

        Returns
            self.mask[(np.asarray(t)) % 30].astype(int) (t): sensitive attribute at t (0/1)
        """
        return self.mask[(np.asarray(t)) % 30].astype(int)

    def _get_reward(
        self,
        p: np.ndarray,
        d: np.ndarray,
    ):
        """Returns reward by multiplying the demand and price vectors
        Args:
            p (np.ndarray): price vector (L)
            d (np.ndarray): demand vector (L)

        Returns:
            p @ d (np.ndarray): revenue from given price and demand vectors
        """
        return p @ d

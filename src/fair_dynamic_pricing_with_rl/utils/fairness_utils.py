import numpy as np


def compute_jains_index(action_hist, groups, p_min, p_diff):
    """
    Compute Jain's fairness index across sensitive-attribute (SNAP-proxied) groups.

    Args:
        action_hist (np.ndarray): raw action indices taken by the policy (T,K,L) or (T,L)
        groups (np.ndarray): sensitive attribute labels (0/1) (T,K) or (T)
        p_min (np.ndarray): minimum prices (L)
        p_diff (np.ndarray) price granularity (L)

    Returns:
        jains_index: float
        group_values: dict {group_label: aggregated_value}
    """
    unique_groups = np.unique(groups)
    group_values = {}

    # handle if action_hist was initialized to be (T,L) shaped
    if action_hist.shape[0] != groups.shape[0]:
        action_hist = action_hist[:groups.shape[0],:]

    # transform action indices to prices based on _action_to_price method
    # rounding guards against floating point inconsistencies
    prices = np.zeros(action_hist.shape)
    prices = np.round(p_min + action_hist * p_diff, 10)

    # basket price per (t, k) or (t), taking only 1-1 product
    total_price = prices.sum(axis=-1)

    for g in unique_groups:
        mask = groups == g
        group_values[g] = total_price[mask].mean()

    x = np.array(list(group_values.values()))
    n = len(x)
    jains_index = (x.sum() ** 2) / (n * np.sum(x**2))

    return jains_index, group_values

import numpy as np


def compute_jains_index(action_hist, groups, p_min, p_diff):
    """
    Compute Jain's fairness index across sensitive-attribute (SNAP-proxied) groups.

    Args:
        action_hist (np.ndarray): raw action indices taken by the policy (T,K,L)
        groups (np.ndarray): sensitive attribute labels (0/1) (T,K)
        p_min (np.ndarray): minimum prices (L)
        p_diff (np.ndarray) price granularity (L)

    Returns:
        jains_index: float
        group_values: dict {group_label: aggregated_value}
    """
    unique_groups = np.unique(groups)
    group_values = {}

    # get main parameters
    T, K, L = action_hist.shape

    # transform action indices to prices based on _action_to_price method
    # rounding guards against floating point inconsistencies
    prices = np.zeros((T, K, L))
    for product_idx in range(L):
        prices[:, :, product_idx] = np.round(
            p_min[product_idx] + action_hist[:, :, product_idx] * p_diff[product_idx],
            10,
        )

    total_price = prices.sum(axis=2)  # basket price per (t, k)

    for g in unique_groups:
        mask = groups == g
        group_values[g] = total_price[mask].mean()

    x = np.array(list(group_values.values()))
    n = len(x)
    jains_index = (x.sum() ** 2) / (n * np.sum(x**2))

    return jains_index, group_values

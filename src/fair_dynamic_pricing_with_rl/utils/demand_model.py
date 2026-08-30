import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import CHOSEN_STATE, PRODUCT_LIST


class DemandModelEstimator:
    def __init__(
        self,
        unit_sales_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        price_granularity: float,
        product_list: list = PRODUCT_LIST,
        chosen_state: str = CHOSEN_STATE,
        verbose: bool = False,
    ):
        """Initialize demand model estimator.

        Args:
            unit_sales_df (pd.DataFrame): table containing information on sales for store-day-product level
            prices_df (pd.DataFrame): table containing prices on week-store-product level
            calendar_df (pd.DataFrame): table containing information on sensitive attribute
            price_granularity (float): price granularity used for price ranges
            product_list (list): list of product IDs to model
            chosen_state (str): chosen state from ["WI", "TX", "CA"]
            verbose (bool, optional): If True, prints OLS summary and price range. (defaults to False)
        """

        # store base data and parameters
        self.unit_sales = unit_sales_df
        self.prices = prices_df
        self.calendar = calendar_df

        self.product_list = product_list
        self.chosen_state = chosen_state
        self.price_gran = price_granularity
        self.verbose = verbose

    def run_pipeline(self):
        """Runs the demand estimation pipeline.

        Returns:
            p_mins (np.ndarray): price minimums (L,)
            p_maxes (np.ndarray): price maximums (L,)
            p_diffs (np.ndarray): price granularity (L,)
            betas (list): estimated parameters for different sensitive attribute values
            max_demand (float): max observed historical demand among the L products
        """

        # create dataframes with price-demand information differently
        # for SNAP and non-SNAP households
        dfs = self._create_price_demand_data()

        # estimate demand models, coefficients are used only
        betas = [
            self._estimate_demand_model(dfs[i], self.product_list)[0]
            for i in range(len(dfs))
        ]

        # get price range
        p_mins, p_maxes, p_diffs = self._get_price_range()

        # get max_demand
        max_demand = np.max(self.pq["quantity"])

        return (p_mins, p_maxes, p_diffs, betas, max_demand)

    def _estimate_demand_model(self, df, product_list: list = PRODUCT_LIST):
        """Estimate model of quantity on log prices (second-order) plus 7-day reference-price gaps using OLS.

        Args:
            df (pd.DataFrame) table containing price and sales data on product-day granularity
            product_list (list) list of product IDs to model (defaults to PRODUCT_LIST)

        Returns:
            pd.DataFrame(params): estimated parameters
            results_dict (dict): fitted smf.ols objects with keys of product IDs
        """

        log_p = self._to_day_index(
            df.pivot_table(
                index="d", columns="item_id", values="log_sell_price", aggfunc="mean"
            )
        )
        q = self._to_day_index(
            df.pivot_table(
                index="d", columns="item_id", values="quantity", aggfunc="mean"
            )
        )

        # lag-1 demand as regressor
        lag_cols = [f"lag1_{col}" for col in q.columns]
        for col in q.columns:
            q[f"lag1_{col}"] = q[col].shift(1)

        X = log_p.add_prefix("log_p_")

        price_cols = list(X.columns)

        mains = " + ".join(f"Q('{c}')" for c in price_cols)
        squares = " + ".join(f"I(Q('{c}')**2)" for c in price_cols)
        lags = " + ".join(f"Q('{c}')" for c in lag_cols)
        formula_rhs = f"({mains})**2 + {squares} + {lags}"

        X = X.join(q[lag_cols])
        params, results_dict = {}, {}

        for product in product_list:
            if self.verbose:
                print(f"--- Fitting demand model for {product} ---")
            data = X.copy()
            data["y"] = q[product]
            data = data.dropna()

            res = smf.ols(f"y ~ {formula_rhs}", data=data).fit()
            params[product] = res.params
            results_dict[product] = res
            if self.verbose:
                print(res.summary())

        return pd.DataFrame(params), results_dict

    def _create_price_demand_data(self):
        """ From raw data tables create price - demand table needed for model estimation.

        Returns:
            [pq_nonsnap, pq_snap]: list of pd.DataFrames with price-demand information on product-day level.
        """

        # Get product-level demand for FOODS per day
        # from EDA, the prices were founds to be state-dependent
        foods_demand = self.unit_sales[self.unit_sales.state_id == self.chosen_state]
        foods_demand = foods_demand.loc[
            foods_demand.cat_id == "FOODS",
            ["item_id"] + [col for col in foods_demand.columns if col.startswith("d_")],
        ]
        q = foods_demand.groupby(by=["item_id"]).mean().reset_index()
        # filter for products
        q = q[q.item_id.isin(self.product_list)]

        merged_df = self.calendar.merge(self.prices, how="left", on="wm_yr_wk")
        merged_df = merged_df[merged_df.item_id.isin(self.product_list)]
        merged_df = merged_df.loc[merged_df.store_id.str.startswith(self.chosen_state)]

        # obtain weekly prices
        p = (
            merged_df.groupby(by=["item_id", "d"])["sell_price"]
            .mean()
            .reset_index()
            .assign(d_num=lambda df: df["d"].str.extract(r"(\d+)").astype(int))
            .sort_values(["item_id", "d_num"])
            .drop(columns="d_num")
        )

        q_long = q.reset_index().melt(
            id_vars="item_id", var_name="d", value_name="quantity"
        )

        pq = p.merge(q_long, how="left", on=["d", "item_id"])
        pq = pq.dropna()

        pq["log_sell_price"] = np.log(pq["sell_price"])

        # add SNAP day flag and create separate data frames
        pq = pq.merge(
            self.calendar[["d", f"snap_{self.chosen_state}"]], how="left", on="d"
        )

        # save price-quantity dataframe
        self.pq = pq

        pq_snap = pq.loc[pq[f"snap_{self.chosen_state}"] == 1].copy()
        pq_nonsnap = pq.loc[pq[f"snap_{self.chosen_state}"] == 0].copy()

        return [pq_nonsnap, pq_snap]

    def _get_price_range(self):
        """Get min. and max. prices of products.

        Returns:
            p_mins (np.ndarray): price minimums (L,)
            p_maxes (np.ndarray): price maximums (L,)
            p_diffs (np.ndarray): price granularity (L,)
        """

        p_mins = np.zeros((len(self.product_list)))
        p_maxes = np.zeros((len(self.product_list)))

        for i in range(len(self.product_list)):
            pq_item = self.pq.loc[self.pq.item_id == PRODUCT_LIST[i]]

            p_mins[i] = np.min(pq_item.sell_price)
            p_maxes[i] = np.max(pq_item.sell_price)

        # the resulting price ranges are printed for testing the env
        if self.verbose:
            print(5 * "-" + " Price minimums " + 5 * "-")
            print(p_mins)
            print(5 * "-" + " Price maximums " + 5 * "-")
            print(p_maxes)

        # price granularity
        p_diffs = self.price_gran * np.ones((len(self.product_list)))

        return (p_mins, p_maxes, p_diffs)

    def _to_day_index(self, frame):
        """ Utility function: d_1, d_2, ... to integer index numerically sorted.
        """
        day = frame.index.str.extract(r"(\d+)", expand=False).astype(int)
        return frame.set_axis(pd.Index(day, name="day")).sort_index()

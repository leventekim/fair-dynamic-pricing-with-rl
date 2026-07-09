import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import PRODUCT_LIST, CHOSEN_STATE, SNAP_BUDGET_RATIO


class AIDSEstimator:
    def __init__(
        self,
        unit_sales_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        price_granularity: float,
        product_list: list = PRODUCT_LIST,
        chosen_state: str = CHOSEN_STATE,
        snap_budget_ratio: float = SNAP_BUDGET_RATIO,
        verbose: bool = False,
    ):
        # store base data and parameters
        self.unit_sales = unit_sales_df
        self.prices = prices_df
        self.calendar = calendar_df

        self.product_list = product_list
        self.chosen_state = chosen_state
        self.snap_budget_ratio = snap_budget_ratio
        self.price_gran = price_granularity
        self.verbose = verbose

    def run_pipeline(self):
        # create dataframes with price-demand information differently
        # for SNAP and non-SNAP households
        dfs = self._create_price_demand_data()

        # estimate aids models, coefficients are used only
        result = [
            self._estimate_aids_model(dfs[i], self.product_list)[0]
            for i in range(len(dfs))
        ]

        # get price range
        p_mins, p_maxes, p_diffs = self._get_price_range()

        # get parameters
        alpha = [result[i][:, 0] for i in range(len(result))]
        gamma = [result[i][:, 1 : (1 + len(PRODUCT_LIST))] for i in range(len(result))]
        beta = [result[i][:, -1] for i in range(len(result))]

        # get budgets
        budgets, real_budgets = self._get_budgets()

        return (p_mins, p_maxes, p_diffs, alpha, gamma, beta, budgets, real_budgets)

    def _estimate_aids_model(self, df, product_list: list = PRODUCT_LIST):
        # Build the wide format log-price matrix
        log_p_wide = df.pivot(
            index="d", columns="item_id", values="log_sell_price"
        ).add_prefix("log_p_")

        # Pull the shared log real-expenditure regressor, one row per period
        period_vars = df.groupby("d")[["log_x_P_star"]].first()

        # Pull shares wide too, so y and X share the same index (d) by construction
        w_wide = df.pivot(index="d", columns="item_id", values="w").add_prefix("w_")

        # joins on the shared 'd' index
        X = log_p_wide.join(period_vars)
        X = sm.add_constant(X)

        price_cols = [c for c in log_p_wide.columns]
        coeff_names = ["const"] + price_cols + ["log_x_P_star"]

        estim_coeff = np.zeros((len(self.product_list), len(coeff_names)))
        results_dict = {}

        for row_i, product in enumerate(self.product_list):
            if self.verbose:
                print(f"--- Fitting AIDS model for {product} ---")

            y = w_wide[f"w_{product}"]

            model = sm.OLS(y, X)
            results = model.fit()
            if self.verbose:
                print(results.summary())

            # save results
            estim_coeff[row_i, :] = results.params.values
            results_dict[product] = results

        return estim_coeff, coeff_names, results_dict

    def _create_price_demand_data(self):
        # get product-level demand for FOODS per day
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

        # calculate share of budget (w_i) on product-level
        pq["revenue"] = pq["sell_price"] * pq["quantity"]
        pq_agg = pq.groupby(by=["d"])["revenue"].sum().reset_index()
        pq_agg = pq_agg.rename(columns={"revenue": "weekly_revenue"})
        pq = pq.merge(pq_agg, how="left", on="d")
        pq["w"] = pq["revenue"] / pq["weekly_revenue"]
        # using notation defined weekly revenue (company's POV) is equivalent to
        # x (household budget)
        pq["log_x"] = np.log(pq["weekly_revenue"])

        # calculate Stone's index
        pq["log_sell_price"] = np.log(pq["sell_price"])
        pq["w_log_sell_price"] = pq["w"] * pq["log_sell_price"]
        pq_agg2 = pq.groupby(by=["d"])["w_log_sell_price"].sum().reset_index()
        pq_agg2 = pq_agg2.rename(columns={"w_log_sell_price": "log_P_star"})
        pq = pq.merge(pq_agg2, how="left", on="d")

        # create explanatory variable from household budget and P*
        pq["log_x_P_star"] = pq["log_x"] - pq["log_P_star"]

        # during log_P_star estimation some NAs were introduced
        na_share = (pq.isna().sum().sum() / pq.shape[0]) * 100
        if self.verbose:
            print(f"The share of NA values: {na_share:.2f}%")
        pq.dropna(inplace=True)

        # add SNAP day flag and create separate data frames
        pq = pq.merge(
            self.calendar[["d", f"snap_{self.chosen_state}"]], how="left", on="d"
        )

        # save price-quantity dataframe
        self.pq = pq

        pq_snap = pq.loc[pq[f"snap_{self.chosen_state}"] == 1].copy()
        pq_nonsnap = pq.loc[pq[f"snap_{self.chosen_state}"] == 0].copy()

        # for SNAP households their budget is scaled by ratio obtained from USDA
        # only done for log_x_P_star as it is used as a regressor
        # note that \log{\frac{\gamma x}{P^*}} = \log{\frac{x}{P^*}} + \log{\gamma}
        pq_snap["log_x_P_star"] = pq_snap["log_x_P_star"] + np.log(SNAP_BUDGET_RATIO)

        return [pq_nonsnap, pq_snap]

    def _get_price_range(self):
        # get min. and max. prices of products
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

    def _get_budgets(self):
        budgets = np.zeros(2)
        real_budgets = np.zeros(2)

        for i in range(2):
            pq_sensitive = self.pq.loc[self.pq.snap_TX == i]

            last_row = np.max(pq_sensitive.index)
            # note here the dual approach used for SNAP households:
            # weekly_revenue (x) is unscaled to ensure consistency with prices
            # while the real budget is scaled to model income effect more closely
            budgets[i] = pq_sensitive.loc[last_row, "weekly_revenue"]
            real_budgets[i] = np.exp(pq_sensitive.loc[last_row, "log_x_P_star"])

        return (budgets, real_budgets)

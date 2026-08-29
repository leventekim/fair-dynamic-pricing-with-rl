# Fair Dynamic Pricing Using Reinforcement Learning

The following is code for the MSc Thesis "Fair Dynamic Pricing Using Reinforcement Learning" of the author submitted in partial fulfillment for the MSc in Machine Learning and Data Science at Imperial College London.

**Author**: Levente KIM

**Date of submission**: 2nd September 2026

## Data Availability

The data used for the project is the M5 Dataset from Kaggle that contains information on price and demand from 10 Walmart stores in the US.

The data can be downloaded from Kaggle's official website from the following link: [M5 - Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data?select=sales_train_evaluation.csv)

Please create a `data` directory and copy all the relevant downloaded files:
- calendar.csv
- sales_train_evaluation.csv
- sell_prices.csv

## Running the codes

Before running the codes to replicate the results, please create an `output` directory and set the FIGURES_PATH parameter in the `src/config.py` file that points to a folder on your local machine.

The orchestrator codes in the `src` folder should be ran in the order as specified in the prefix of their names:
1. 00_run_eda.ipynb (Exploratory data analysis)
2. 01_run_experiment_1.ipynb (Experiment 1 - taking a random action with some probability)
3. 02_run_experiment_2.ipynb (Experiment 2 - incorporating Jain's index into the reward function)
4. 03_run_experiment_3.ipynb (Experiment 3 - masking the sensitive attribute that is received by the algorithm)
5. 04_compare_experiments.ipynb (Comparison of experiments for the final report)

**Runtime**: approx. 12-14 hours on a Macbook Pro, M3 Pro with macOS 26.3.

## Replicability of Supplementary Material Graphs

Supplementary Material graphs for exploratory data analysis are replicable by running the `src/00_run_eda.ipynb` script. The graphs of the Hyperparameter Tuning sub-section are replicable by setting ent_coef to 0.05 and re-running the experiments. (steps 2-4) Finally, additional Experiment 2 plots can be recreated by the corresponding scripts in the `notebooks/` directory with a suffix indicating the change in the fairness reward component. 
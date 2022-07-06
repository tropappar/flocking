#!/bin/python3

import pandas as pd
import numpy as np
from mesa.batchrunner import BatchRunner
from mesa.datacollection import DataCollector

from swarm.model import Swarm


params_variable = {
    "population": range(10, 101, 10),
}

params_fixed = {
    "interaction_range":      30,
    "debug":                  False,
    "targets":                1,
    "show_interaction_range": False,
    "vision_range":           1,
    "show_vision_range":      False,
    "help_range":             1,
    "min_agents":             1,
    "max_agents":             1,
}


if __name__ == "__main__":
    # repetitions of each setup
    iterations = 3

    # run simulations
    batch_run = BatchRunner(Swarm, params_variable, params_fixed, iterations=iterations, max_steps=250, model_reporters={"Data Collector": lambda m: m.datacollector})
    batch_run.run_all()

    # get data of simulation runs
    # beware of this bug: column headers are sometimes wrong: https://github.com/projectmesa/mesa/issues/877
    df_raw = batch_run.get_model_vars_dataframe()

    # list of parameters
    params = list(params_variable.keys())

    # write raw data to csv files
    # df_raw.to_csv("data/runs.csv")
    # for i,d in enumerate(df_raw["Data Collector"]):
        # d.get_model_vars_dataframe().to_csv("data/run_{0}.csv".format(i))

    # aggregate steps of each run
    rows = []
    for row in df_raw.iterrows():
        # extract measurements
        row_df = row[1]["Data Collector"].get_model_vars_dataframe()

        # construct new row
        index = params + list(row_df.columns)
        cols_params = row[1][params]
        cols_meas = row_df.mean() # average measurements
        new_row = pd.Series(pd.concat([cols_params,cols_meas]), index=index, name=row[0])
        rows.append(new_row)
    df_agg = pd.DataFrame(rows)

    # aggregate runs with same parameters
    df_agg = df_agg.groupby(params).mean()

    # write aggregated data to csv file
    # df_agg.to_csv("data/aggregated.csv")

    # write number of messages to text files
    df_agg["Messags Centralized"].to_csv("data/abstract_messages_centralized_{0}.txt".format(30), sep=" ", index_label=["n"], header=["m"])

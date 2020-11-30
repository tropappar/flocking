#!/bin/python3

import pandas as pd
import numpy as np
from mesa.batchrunner import BatchRunner
from mesa.datacollection import DataCollector

from flocking.model import BoidFlockers


params_variable = {
    "vision":         range(6, 61, 6),
    "population":     range(2, 149, 10),
}

params_fixed = {
    "formation":      "Grid",
    "width":          100,
    "height":         100,
    "min_dist":       1,
    "flock_vel":      1,
    "accel_time":     1,
    "equi_dist":      5,
    "repulse_max":    10,
    "repulse_spring": 0.5,
    "align_frict":    0.3,
    "align_slope":    30,
    "align_min":      0.1,
    "wall_decay":     10,
    "wall_frict":     1,
    "form_shape":     1,
    "form_track":     1,
    "form_decay":     10,
    "wp_tolerance":   10,
}


if __name__ == "__main__":
    # repetitions of each setup
    iterations = 3

    # run simulations
    batch_run = BatchRunner(BoidFlockers, params_variable, params_fixed, iterations=iterations, max_steps=250, model_reporters={"Data Collector": lambda m: m.datacollector})
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
    for r in params_variable["vision"]:
        df_n = df_agg.loc[r]
        df_n["Messags Centralized"].to_csv("data/abstract_messages_centralized_{0}.txt".format(r), sep=" ", index_label=["n"], header=["m"])

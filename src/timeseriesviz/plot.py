import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import math
import json


def plot_timeseries(Y, pred, title, splitsize=6, save_path=None):

    # Y is the actual validation data and pred is the predicted data
    figure = plt.figure()
    totalDays = Y.shape[0]
    Nloc = Y.shape[1]

    Numpoints = math.floor((totalDays+0.001)/splitsize)
    neededrows = math.floor(splitsize/2)

    plt.rcParams["figure.figsize"] = [24,36]
    figure, axs = plt.subplots(nrows=neededrows+1, ncols=2)

    main_plot = axs[0,0]

    # Calculate error
    error = Y - pred

    main_plot.plot(np.arange(0, totalDays), np.sum(Y/Nloc, axis=1), label=f'real')
    main_plot.plot(np.arange(0, totalDays), np.sum(pred/Nloc, axis=1), label='prediction')
    main_plot.plot(np.arange(0, totalDays), np.sum(error/Nloc, axis=1), label=f'error', color="red")

    main_plot.set_title(title, fontsize=20)

    main_plot.set_ylabel('y', color="black", fontweight='bold', fontsize=18)
    main_plot.set_xlabel('time', color="black",fontweight='bold', fontsize=18)

    main_plot.grid(False)

    # Simplified legend location
    main_plot.legend(fontsize=16, loc = 'best')

    axs[0, 1].set_visible(False)


    plt_count = 0
    for nrow in range(neededrows):
        for i_plot in range(2):
            eachplt = axs[nrow+1, i_plot]

            # Calculate error and sum of error
            error = Y - pred

            start_i = plt_count * Numpoints
            end_i = start_i + Numpoints

            eachplt.plot(np.arange(0, Numpoints), np.sum(Y[start_i : end_i, :]/Nloc, axis=1), label=f'real')
            eachplt.plot(np.arange(0, Numpoints), np.sum(pred[start_i : end_i, :]/Nloc, axis=1), label='prediction')
            eachplt.plot(np.arange(0, Numpoints), np.sum(error[start_i : end_i, :]/Nloc, axis=1), label=f'error', color="red")


            eachplt.set_title(f"Detailed plot: {start_i} to {end_i}", fontsize=20)

            eachplt.set_ylabel('y', color="black", fontweight='bold', fontsize=18)
            eachplt.set_xlabel('time', color="black", fontweight='bold', fontsize=18)

            eachplt.grid(False)

            eachplt.legend(fontsize = 16, loc = 'best')

            plt_count += 1


    figure.tight_layout()
    if save_path:
        plt.savefig(save_path)

    plt.show()


    return figure, axs



def plot_neuralforecast(y_df, model_name, title, splitsize=6, save_path=None):
    # Sanity check
    required = ['unique_id', 'ds', 'y']
    missing = [c for c in required if c not in y_df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    Nloc = y_df['unique_id'].unique().size
    totalDays = y_df['ds'].unique().size

    y_np = y_df['y'].to_numpy()
    pred_np = y_df[model_name].to_numpy()
    y = np.reshape(y_np, (Nloc, totalDays))
    y = np.transpose(y)

    pred = np.reshape(pred_np, (Nloc, totalDays))
    pred = np.transpose(pred)

    fig, axs = plot_timeseries(y[:,:], pred[:,:], title, splitsize, save_path)
    
    return fig, axs



def plot_numpy(y, pred, title, splitsize=6, save_path=None):
    # Sanity check
    if y.ndim != 2 or pred.ndim != 2:
        raise ValueError(f"y and pred must be 2D arrays [time x unique_id]; got y.ndim={y.ndim}, pred.ndim={pred.ndim}.")
    
    if y.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y{y.shape} vs pred{pred.shape}.")
    
    if y.size == 0:
        raise ValueError("Empty arrays provided.")

    fig, axs = plot_timeseries(y, pred, title, title, splitsize, save_path)

    return fig, axs
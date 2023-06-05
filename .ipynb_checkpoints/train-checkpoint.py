import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae


def Train(
    df_train,
    test_size = 0.2,
    col = "cnt",
    params = {
        "objective": "regression",
        "metrics": "mae",
    },
    early_stopping_rounds = 100,
    itern = 100
):
    model_stock = list()

    for i in range(0,itern):
        df_trn, df_val = train_test_split(df_train, test_size=test_size)

        train_y = df_trn[col]
        train_x = df_trn.drop(col, axis=1)

        val_y = df_val[col]
        val_x = df_val.drop(col, axis=1)

        trains = lgb.Dataset(train_x, train_y)
        valids = lgb.Dataset(val_x, val_y)

        model = lgb.train(
            params, 
            trains,
            valid_sets=valids,
            num_boost_round=1000,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )

        model_stock.append(model)
        print("training No.{} finished!\n".format(i+1))

    return  model_stock
   

def EvalTrain(pred_stock, actl_vals):
    mae_list = np.array([])
    for pred in pred_stock:
        mae_list = np.append( mae_list, mae(actl_vals, pred) )

    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    
    return (mae_mean, mae_std)

def EvalTrain(pred_vals, actl_vals):
    acc = mae(pred_vals, actl_vals)
    
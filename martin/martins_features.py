from xgboost import XGBRegressor
import pandas as pd 
import numpy as np
from sklearn.model_selection import  LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import torch
from sklearn.ensemble import RandomForestRegressor
import gc
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

#import labeled features and labeled demographics 
data = pd.read_csv(r"D:\Martin\For Corin\Data\all_data_labeled_interpolated.csv")
data["timestamp"] = pd.to_datetime(data["timestamp"])
data.set_index(data["timestamp"], inplace=True)
data = data.resample("5T").mean()
data = data.drop(columns=['ID'])
data.to_csv(r"D:\Vittoria\Code\data\martin\only_martins_features.csv", index = False)

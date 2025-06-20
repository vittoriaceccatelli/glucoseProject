import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")

z = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))

threshold = 3
data = data[(z < threshold).all(axis=1)]
data.to_csv(r"D:\Vittoria\Code\data\other\labeled_features_no_outliers.csv", index=False)
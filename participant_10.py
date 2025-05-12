import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

features_path = r'D:\Vittoria\Code\data'

#plots all features for all participants
file_path = rf'{features_path}\labeled_features_no_outliers.csv'
data = pd.read_csv(file_path)

data_part = data[data["Participant_ID"] == 10]
data_par = data[data["Participant_ID"] == 11]
data_part = data_part.drop(columns=['Participant_ID'])
data_par = data_par.drop(columns=['Participant_ID'])

for column in data_part.columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    sns.boxplot(data=data_part[column], ax=axes[0])
    sns.boxplot(data=data_par[column], ax=axes[1])
    plt.title(column)
    plt.tight_layout() 
    plt.show()
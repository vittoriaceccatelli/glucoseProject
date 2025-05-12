import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import torch

#GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

features_path = r'D:\Vittoria\Code\data'

#import labeled features and demographics
#file_path = rf'{features_path}\martin\martins_features.csv'
file_path = rf'{features_path}\martin\labeled_features_no_correlation_martin_pers.csv'
data = pd.read_csv(file_path)

data['Time'] = pd.to_datetime(data['Time'])

df_list = []

for part in range(1, 17):
    data_part = data[data["Participant_ID"] == part]

    part_data = {
        "Participant_ID": part,
        "PersNorm_before_balancing": (data_part["PersStatus"] == "PersNorm").sum(),
        "PersHigh_before_balancing": (data_part["PersStatus"] == "PersHigh").sum(),
        "PersLow_before_balancing": (data_part["PersStatus"] == "PersLow").sum(),
    }

    df_list.append(part_data)

df = pd.DataFrame(df_list)
#df.to_csv(r"D:\Vittoria\Code\data\persClasses_before_balancing.csv")

#separate DataFrame into different classes
df_persnorm = data[data['PersStatus'] == 'PersNorm']
df_pershigh = data[data['PersStatus'] == 'PersHigh']
df_perslow = data[data['PersStatus'] == 'PersLow']

#select length of shortest dataframe of the three
target_n_samples = min([len(df_persnorm), len(df_pershigh), len(df_perslow)])

#randomly drop datapoints from the dataframes so that in the end all three dataframes have the same amount of datapoints 
df_persnorm_sampled = df_persnorm.sample(n=target_n_samples, random_state=42)
df_pershigh_sampled = df_pershigh.sample(n=target_n_samples, random_state=42)
df_perslow_sampled = df_perslow.sample(n=target_n_samples, random_state=42)
df_balanced = pd.concat([df_persnorm_sampled, df_pershigh_sampled, df_perslow_sampled])
df_balanced = df_balanced.sort_values(by=['Participant_ID', 'Time']).reset_index(drop=True)

#merge features and demographics data and save balanced features+demographics information
#df_balanced.to_csv(r'data\martin\balanced_dataset_martin.csv', index=False)
df_balanced.to_csv(r'data\martin\balanced_dataset_martin_no_correlation.csv', index=False)


df_list = []

for part in range(1, 17):
    data_part = df_balanced[df_balanced["Participant_ID"] == part]

    part_data = {
        "Participant_ID": part,
        "PersNorm_before_balancing": (data_part["PersStatus"] == "PersNorm").sum(),
        "PersHigh_before_balancing": (data_part["PersStatus"] == "PersHigh").sum(),
        "PersLow_before_balancing": (data_part["PersStatus"] == "PersLow").sum(),
    }

    df_list.append(part_data)

df = pd.DataFrame(df_list)
#df.to_csv(r"D:\Vittoria\Code\data\persClasses_after_balancing.csv")
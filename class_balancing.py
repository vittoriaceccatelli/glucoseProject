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

dataset = input("Labeled or NoCorrelation? ")
continuous = input("ContinuousYes or ContinuousNo? ")

#import labeled features and demographics
if dataset=="Labeled":
    file_path = rf'{features_path}\labeled_features.csv'
elif dataset=="NoCorrelation":
    file_path = rf'{features_path}\labeled_features_no_correlation_RandomForest_Classification.csv'

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
#df.to_csv(rf"D:\Vittoria\Code\data\persClasses_before_balancing_{dataset}.csv")

#separate DataFrame into different classes
df_persnorm = data[data['PersStatus'] == 'PersNorm']
df_pershigh = data[data['PersStatus'] == 'PersHigh']
df_perslow = data[data['PersStatus'] == 'PersLow']

#select length of shortest dataframe of the three
target_n_samples = min([len(df_persnorm), len(df_pershigh), len(df_perslow)])

#randomly drop datapoints from the dataframes so that in the end all three dataframes have the same amount of datapoints
df_pershigh_sampled = df_pershigh.sample(n=target_n_samples, random_state=42)
df_perslow_sampled = df_perslow.sample(n=target_n_samples, random_state=42)
if continuous=="ContinuousNo": 
    df_persnorm_sampled = df_persnorm.sample(n=target_n_samples, random_state=42)
    df_balanced = pd.concat([df_persnorm_sampled, df_pershigh_sampled, df_perslow_sampled])

    df_balanced = df_balanced.sort_values(by=['Participant_ID', 'Time']).reset_index(drop=True)

    df_balanced.to_csv(rf'data\balanced_dataset_random_{dataset}.csv', index=False)

elif continuous=="ContinuousYes":
    drop = 0
    drop_limit = len(df_persnorm) - target_n_samples
    df_persnorm_sampled = pd.DataFrame()

    for part in range(1, 17):
        data_part = data[data["Participant_ID"] == part]
        current_sequence = 0
        indices_to_drop = []
        kept_indices = []

        for index, row in data_part.iterrows():
            if row["PersStatus"] == "PersNorm":
                current_sequence += 1 
                indices_to_drop.append(index)
            else:
                if current_sequence > 12:
                    if drop + current_sequence <= drop_limit:
                        drop += current_sequence
                        kept_indices.extend(indices_to_drop)

                current_sequence = 0
                indices_to_drop = []

        if current_sequence > 12:
            if drop + current_sequence <= drop_limit:
                drop += current_sequence
                kept_indices.extend(indices_to_drop)

        data_part = data_part.drop(kept_indices)

        df_persnorm_sampled = pd.concat([df_persnorm_sampled, data_part])

    df_persnorm_sampled = df_persnorm_sampled[df_persnorm_sampled["PersStatus"] == "PersNorm"]

    df_balanced = pd.concat([df_persnorm_sampled, df_pershigh_sampled, df_perslow_sampled])

    df_balanced = df_balanced.sort_values(by=['Participant_ID', 'Time']).reset_index(drop=True)

    df_balanced.to_csv(rf'data\balanced_dataset_more_continuous_{dataset}.csv', index=False)

print(df_balanced["PersStatus"].value_counts())




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
if continuous=="ContinuousNo": 
    df.to_csv(rf"D:\Vittoria\Code\data\persClasses_after_balancing_random_{dataset}.csv")
elif continuous=="ContinuousYes":
    df.to_csv(rf"D:\Vittoria\Code\data\persClasses_after_balancing_more_continuous_{dataset}.csv")
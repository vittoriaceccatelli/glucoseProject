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
file_path = rf'{features_path}\labeled_features_zdenek_1s.csv'
data = pd.read_csv(file_path)

#if in columns 2 to 42 (excluding food columns where 0 often appears) there are more than 35 zeros --> drop row 
#--> wearable device was not recording data
data.rename(columns={"index": "Time"}, inplace=True)
data['Time'] = pd.to_datetime(data['Time'])
data = data.dropna()

demo = rf'D:\Vittoria\Code\data\demographics.csv'
demo_data = pd.read_csv(demo)

#assign 0 to males and 1 to females
demo_data['Gender'] = demo_data['Gender'].map({'MALE': 0, 'FEMALE': 1})

#separate DataFrame into different classes
df_persnorm = data[data['PersStatus'] == 'PersNorm']
df_pershigh = data[data['PersStatus'] == 'PersHigh']
df_perslow = data[data['PersStatus'] == 'PersLow']

#select length of shortest dataframe of the three
target_n_samples = min([len(df_persnorm), len(df_pershigh), len(df_perslow)])

#randomly drop datapoints from the dataframes so that in the end all three dataframes have the same amount of datapoints 
#CORIN SUGGESTED TO DROP maybe a whole participant with a lot of persnorms etc ... but how?
df_persnorm_sampled = df_persnorm.sample(n=target_n_samples, random_state=42)
df_pershigh_sampled = df_pershigh.sample(n=target_n_samples, random_state=42)
df_perslow_sampled = df_perslow.sample(n=target_n_samples, random_state=42)
df_balanced = pd.concat([df_persnorm_sampled, df_pershigh_sampled, df_perslow_sampled])
df_balanced = df_balanced.sort_values(by=['Participant_ID', 'Time']).reset_index(drop=True)

df_balanced = df_balanced.merge(demo_data, left_on='Participant_ID', right_on='ID', how='left')

#merge features and demographics data and save balanced features+demographics information
df_balanced.to_csv(r'data\balanced_dataset_zdenek_1s.csv', index=False)


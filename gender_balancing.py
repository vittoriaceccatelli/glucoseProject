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
    data = pd.read_csv(file_path)
    demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
    data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
    data['Time'] = pd.to_datetime(data['Time'])

    #separate DataFrame into different classes
    males = data[data['Gender'] == 'MALE']
    females = data[data['Gender'] == 'FEMALE']
elif dataset=="NoCorrelation":
    file_path = rf'{features_path}\labeled_features_no_correlation_RandomForest_Classification.csv'
    data = pd.read_csv(file_path)
    data['Time'] = pd.to_datetime(data['Time'])

    #separate DataFrame into different classes
    males = data[data['Gender'] == 0]
    females = data[data['Gender'] == 1]



#select length of shortest dataframe of the three
target_n_samples = min([len(males), len(females)])

if continuous=="ContinuousNo": 
    df_males_sampled = males.sample(n=target_n_samples, random_state=42)
    df_females_sampled = females.sample(n=target_n_samples, random_state=42)

    df_balanced = pd.concat([df_males_sampled, df_females_sampled])

    df_balanced = df_balanced.sort_values(by=['Participant_ID', 'Time']).reset_index(drop=True)

    df_balanced.to_csv(rf'data\gender_balanced_dataset_random_{dataset}.csv', index=False)
    
    print(df_balanced['Gender'].value_counts())
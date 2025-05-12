import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, r2_score, confusion_matrix, make_scorer, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#import torch
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import gc

#GPU
gc.collect()
#torch.cuda.empty_cache()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

#separate features and labels 
#dataset = input("Oversampled, BalancedRandom or BalancedContinuous? ")
dataset = 'Oversampled'
food = input("foodYes or foodNo? ")
correlation = input("CorrelationYes or CorrelationNo? ")

np.random.seed(42)

if dataset=="BalancedRandom":
    if correlation=="CorrelationYes":
        df_balanced = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_random_Labeled.csv")
        demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
        df_balanced = pd.merge(df_balanced, demo, left_on="Participant_ID", right_on="ID")
        X = df_balanced.drop(columns=['Participant_ID', 'Time', 'PersStatus', 'Glucose_Mean', 'DiabStatus'])
    elif correlation=="CorrelationNo":
        df_balanced = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_random_NoCorrelation.csv")
        X = df_balanced.drop(columns=['Participant_ID', 'Time', 'PersStatus', 'Glucose_Mean'])
    X['Gender'] = X['Gender'].map({'MALE': 0, 'FEMALE': 1})
elif dataset=="BalancedContinuous":
    if correlation=="CorrelationYes":
        df_balanced = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_more_continuous_Labeled.csv")
        demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
        df_balanced = pd.merge(df_balanced, demo, left_on="Participant_ID", right_on="ID")
        df_balanced['Gender'] = df_balanced['Gender'].map({'MALE': 0, 'FEMALE': 1})
        X = df_balanced.drop(columns=['Participant_ID', 'Time', 'PersStatus', 'Glucose_Mean', 'DiabStatus'])
    elif correlation=="CorrelationNo":
        df_balanced = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_more_continuous_NoCorrelation.csv")
        df_balanced['Gender'] = df_balanced['Gender'].map({'MALE': 0, 'FEMALE': 1})
        X = df_balanced.drop(columns=['Participant_ID', 'Time', 'PersStatus', 'Glucose_Mean'])
elif dataset=="Oversampled":
    if correlation=="CorrelationYes":
        df_balanced = pd.read_csv(r"D:\Vittoria\Code\data\oversampled_Labeled.csv")
        demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
        df_balanced = pd.merge(df_balanced, demo, left_on="Participant_ID", right_on="ID")
        df_balanced['Gender'] = df_balanced['Gender'].map({'MALE': 0, 'FEMALE': 1})
        X = df_balanced.drop(columns=['Participant_ID', 'PersStatus', 'DiabStatus'])
    elif correlation=="CorrelationNo":
        df_balanced = pd.read_csv(r"D:\Vittoria\Code\data\oversampled_NoCorrelation.csv")
        df_balanced['Gender'] = df_balanced['Gender'].map({'MALE': 0, 'FEMALE': 1})
        X = df_balanced.drop(columns=['Participant_ID', 'PersStatus'])
    test_fold = [-1] * len(X)
    for i in range(1000):
        test_fold[i] = 0  


#drop food columns
if food=="foodNo":
    if correlation=="CorrelationYes":
        X = X.drop(columns=X.columns[47:66])
    elif correlation=="CorrelationNo":
        X = X.drop(columns=X.columns[24:29])

y = df_balanced['PersStatus'].map({'PersNorm': 0, 'PersHigh': 1, 'PersLow': 2})










dt_model = DecisionTreeClassifier(random_state=42)
accuracy_list, precision_list, recall_list, f1_list, r2_list = [], [], [], [], []
for i in range (10):
    test_index = np.random.choice(26768, size=int(len(X)*(1/3)), replace=False)
    train_index = np.setdiff1d(np.arange(len(X)), test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)

    # Compute metrics
    accuracy_list.append(balanced_accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred, average="weighted"))
    recall_list.append(recall_score(y_test, y_pred, average="weighted"))
    f1_list.append(f1_score(y_test, y_pred, average="weighted"))
    r2_list.append(r2_score(y_test, y_pred))

    report = classification_report(y_test, y_pred)

# Compute mean and standard deviation for each metric
print("Decision Tree Classifier (70/30 split)")
print(f"Balanced Accuracy: {np.mean(accuracy_list):.3f} ± {np.std(accuracy_list):.3f}")
print(f"Precision: {np.mean(precision_list):.3f} ± {np.std(precision_list):.3f}")
print(f"Recall: {np.mean(recall_list):.3f} ± {np.std(recall_list):.3f}")
print(f"F1 Score: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")
print()

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['PersNorm', 'PersHigh', 'PersLow'], yticklabels=['PersNorm', 'PersHigh', 'PersLow'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Decision Tree")
#plt.show()













#setup repeated stratified k fold cross validator, 10 splits with always 9 splits as train and one split as test and
#repeat 3 times 
# Define Repeated Stratified KFold cross-validation (train and test)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

pipeline = Pipeline([
    ('feature_selection', RFE(DecisionTreeClassifier(random_state=42), n_features_to_select=20)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

scoring = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'precision': make_scorer(precision_score, average="weighted"),
    'recall': make_scorer(recall_score, average="weighted"),
    'f1': make_scorer(f1_score, average="weighted")
}

balanced_accuracies = []
precisions = []
recalls = []
f1_scores = []

for i in range(3):
    test_index = np.random.choice(26768, size=len(X)//10, replace=False)
    train_index = np.setdiff1d(np.arange(len(X)), test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average="weighted"))
    recalls.append(recall_score(y_test, y_pred, average="weighted"))
    f1_scores.append(f1_score(y_test, y_pred, average="weighted"))

print("Primary Cross Validated model with repeated stratified k fold")
print(f"Balanced Accuracy: {np.mean(balanced_accuracies):.3f} ± {np.std(balanced_accuracies):.3f}")
print(f"Precision Score: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
print(f"Recall Score: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
print(f"F1 Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")




























'''lr_model = LogisticRegression(max_iter=1000, random_state=42)
accuracy_list, precision_list, recall_list, f1_list, r2_list = [], [], [], [], []
for i in range (10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Compute metrics
    accuracy_list.append(balanced_accuracy_score(y_test, y_pred_lr))
    precision_list.append(precision_score(y_test, y_pred_lr, average="weighted"))
    recall_list.append(recall_score(y_test, y_pred_lr, average="weighted"))
    f1_list.append(f1_score(y_test, y_pred_lr, average="weighted"))
    r2_list.append(r2_score(y_test, y_pred_lr))

    report = classification_report(y_test, y_pred_lr)

# Compute mean and standard deviation for each metric
print("Logistic Regression Metrics (Mean ± Std Dev)")
print(f"Balanced Accuracy: {np.mean(accuracy_list):.3f} ± {np.std(accuracy_list):.3f}")
print(f"Precision: {np.mean(precision_list):.3f} ± {np.std(precision_list):.3f}")
print(f"Recall: {np.mean(recall_list):.3f} ± {np.std(recall_list):.3f}")
print(f"F1 Score: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")'''

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, r2_score, confusion_matrix, make_scorer, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.feature_selection import RFE
import time
from sklearn.pipeline import Pipeline
import gc
import torch 

#GPU
gc.collect()
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#separate features and labels 
#df_balanced = pd.read_csv(r"D:\Vittoria\Code\data\martin\balanced_dataset_martin.csv")
df_balanced = pd.read_csv(r"D:\Vittoria\Code\data\martin\balanced_dataset_martin_no_correlation.csv")
X = df_balanced.drop(columns=['Participant_ID', 'Time', 'PersStatus', 'Glucose_Mean'])

y = df_balanced['PersStatus'].map({'PersNorm': 0, 'PersHigh': 1, 'PersLow': 2})









#setup repeated stratified k fold cross validator, 10 splits with always 9 splits as train and one split as test and
#repeat 3 times 
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

#select 20 features
pipeline = Pipeline([
    ('feature_selection', RFE(DecisionTreeClassifier(random_state=42), n_features_to_select=20)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

#define scoring metrics
scoring = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'precision': make_scorer(precision_score, average="weighted"),
    'recall': make_scorer(recall_score, average="weighted"),
    'f1': make_scorer(f1_score, average="weighted"),
}

#perform cross-validation with scores defined above
cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

#compute mean and standard deviation for each metric
print("Primary Cross Validated model with repeated stratified k fold")
print(f"Balanced Accuracy: {np.mean(cv_results['test_balanced_accuracy']) * 100:.2f}% ± {np.std(cv_results['test_balanced_accuracy']) * 100:.2f}%")
print(f"Precision Score: {np.mean(cv_results['test_precision']) * 100:.2f}% ± {np.std(cv_results['test_precision']) * 100:.2f}%")
print(f"Recall Score: {np.mean(cv_results['test_recall']) * 100:.2f}% ± {np.std(cv_results['test_recall']) * 100:.2f}%")
print(f"F1 Score: {np.mean(cv_results['test_f1']) * 100:.2f}% ± {np.std(cv_results['test_f1']) * 100:.2f}%")
print()







dt_model = DecisionTreeClassifier(random_state=42)
accuracy_list, precision_list, recall_list, f1_list, r2_list = [], [], [], [], []
for i in range (10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
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
print(f"Balanced Accuracy: {np.mean(accuracy_list) * 100:.2f}% ± {np.std(accuracy_list) * 100:.2f}%")
print(f"Precision: {np.mean(precision_list) * 100:.2f}% ± {np.std(precision_list) * 100:.2f}%")
print(f"Recall: {np.mean(recall_list) * 100:.2f}% ± {np.std(recall_list) * 100:.2f}%")
print(f"F1 Score: {np.mean(f1_list) * 100:.2f}% ± {np.std(f1_list) * 100:.2f}%")
print()

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['PersNorm', 'PersHigh', 'PersLow'], yticklabels=['PersNorm', 'PersHigh', 'PersLow'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Decision Tree")
#plt.show()









lr_model = LogisticRegression(max_iter=1000, random_state=42)
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
print(f"Balanced Accuracy: {np.mean(accuracy_list) * 100:.2f}% ± {np.std(accuracy_list) * 100:.2f}%")
print(f"Precision: {np.mean(precision_list) * 100:.2f}% ± {np.std(precision_list) * 100:.2f}%")
print(f"Recall: {np.mean(recall_list) * 100:.2f}% ± {np.std(recall_list) * 100:.2f}%")
print(f"F1 Score: {np.mean(f1_list) * 100:.2f}% ± {np.std(f1_list) * 100:.2f}%")

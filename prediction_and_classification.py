from xgboost import XGBRegressor
import pandas as pd 
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
import gc
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

gc.collect()

#GPU
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def classify_glucose(df):
    #first assign all datapoints PersNorm status
    df['PersStatus'] = 'PersNorm'

    #calculate mean and std of last day, for first datapoints smaller window is used 
    rolling_mean = df["Glucose"].rolling(window=288, min_periods=1).mean()
    rolling_std = df["Glucose"].rolling(window=288, min_periods=1).std()

    #calculate mean + one std and mean - one std
    highThreshold = rolling_mean + rolling_std
    lowThreshold = rolling_mean - rolling_std

    #assign pershigh if higher than highthreshold and perslow is lower than lowthreshold 
    df.loc[df["Glucose"] > highThreshold, 'PersStatus'] = 'PersHigh'
    df.loc[df["Glucose"] < lowThreshold, 'PersStatus'] = 'PersLow'
    return df

correlation = "Labeled"
food = "foodYes"
dataset = "BalancedRandom"

if dataset=="BalancedRandom":
    if correlation=="Labeled":
        data = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_random_Labeled.csv")
        demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
        data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
        data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})
        feature_columns = [col for col in data.columns if col not in ['PersStatus', 'ID', 'Glucose_Mean', 'Time', 'DiabStatus']]
        if food=="foodNo":
            #drop food log columns
            feature_columns = feature_columns[:48] + feature_columns[67:]
    elif correlation=="NoCorrelation":
        data = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_random_NoCorrelation.csv")
        data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})
        feature_columns = [col for col in data.columns if col not in ['PersStatus', 'ID', 'Glucose_Mean', 'Time', 'DiabStatus']]
        if food=="foodNo":
            #drop food log columns
            feature_columns = feature_columns[:26] + feature_columns[30:]
elif dataset=="BalancedContinuous":
    if correlation=="Labeled":
        data = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_more_continuous_Labeled.csv")
        demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
        data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
        data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})
        feature_columns = [col for col in data.columns if col not in ['PersStatus', 'ID', 'Glucose_Mean', 'Time', 'DiabStatus']]
        if food=="foodNo":
            #drop food log columns
            feature_columns = feature_columns[:48] + feature_columns[67:]
    elif correlation=="NoCorrelation":
        data = pd.read_csv(r"D:\Vittoria\Code\data\balanced_dataset_more_continuous_NoCorrelation.csv")
        data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})
        feature_columns = [col for col in data.columns if col not in ['PersStatus', 'ID', 'Glucose_Mean', 'Time', 'DiabStatus']]
        if food=="foodNo":
            #drop food log columns
            feature_columns = feature_columns[:26] + feature_columns[30:]
elif dataset == "nothing": 
    #import labeled features and labeled demographics 
    if correlation=="Labeled":
        data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")
        demo = pd.read_csv(r"D:\Vittoria\Code\data\labeled_demographics.csv")
        data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")
        data['Gender'] = data['Gender'].map({'MALE': 0, 'FEMALE': 1})
        #get names of feature columns and divide data into features and labels and get group IDs (participant IDs)
        feature_columns = [col for col in data.columns if col not in ['PersStatus', 'ID', 'Glucose_Mean', 'Time', 'DiabStatus']]
        if food=="foodNo":
            #drop food log columns
            feature_columns = feature_columns[:48] + feature_columns[67:]
    elif correlation=="NoCorrelation":
        data = pd.read_csv(r'D:\Vittoria\Code\data\labeled_features_no_correlation_XGBoost_Prediction.csv')
        #get names of feature columns and divide data into features and labels and get group IDs (participant IDs)
        feature_columns = [col for col in data.columns if col not in ['PersStatus', 'ID', 'Glucose_Mean', 'Time', 'DiabStatus']]
        if food=="foodNo":
            #drop food log columns
            feature_columns = feature_columns[:26] + feature_columns[30:]


X = data[feature_columns]
y = data[['Glucose_Mean', 'Participant_ID']]

groups = data['Participant_ID']

#define parameters to be used for XGBRegressor
xgb_params = {
    "max_depth": 6,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": 42
}


#PERSONALIZED MODEL 
predictions_df = []
xtest_df = []
glucose_by_status = []

for participant_id in groups.unique():
    #get participant data
    participant_data = X[X["Participant_ID"] == participant_id]
    y_part = y[y["Participant_ID"] == participant_id]
    y_part = y_part.drop(columns=['Participant_ID'])

    #divide data into two parts, first half is training data and second half is test data
    mid_point = len(participant_data) // 2
    train_data = participant_data.iloc[:mid_point]
    test_data = participant_data.iloc[mid_point:]
    y_train =  y_part.iloc[:mid_point]
    y_test=  y_part.iloc[mid_point:]

    #fit random forest regressor for feature importance
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(train_data, y_train)

    #select features with an importance of over 0.005
    feature_importances = pd.Series(rf.feature_importances_, index=feature_columns)
    selected_features = feature_importances[feature_importances > 0.005].index.tolist()

    #get data from selected features
    X_train, y_train = train_data[selected_features], y_train
    X_test, y_test = test_data[selected_features], y_test

    #train xgb Regressor with specified parameters and fit model
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    #INSERT ASSIGNING OF PERS LABELS

    #predict labels from testing data
    pred_labels = pd.DataFrame(model.predict(X_test))
    y_train = pd.DataFrame(y_train)
    labels = pd.concat([y_train, pred_labels], axis=0)
    labels = labels.fillna(0)
    labels["Glucose"] = labels["Glucose_Mean"] + labels[0]
    labels = labels.drop(columns=["Glucose_Mean", 0])
    labels = labels.reset_index(drop=True)
    participant_data = participant_data.reset_index(drop=True)
    feat_and_labels = pd.concat([participant_data, labels], axis=1)

    classified = classify_glucose(feat_and_labels)

        



    X_class = classified.drop(columns=["Glucose", "PersStatus"])
    y_class = classified["PersStatus"]


    #setup repeated stratified k fold cross validator, 10 splits with always 9 splits as train and one split as test and
    #repeat 3 times 
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

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
    cv_results = cross_validate(pipeline, X_class, y_class, cv=cv, scoring=scoring)

    #compute mean and standard deviation for each metric
    print("Primary Cross Validated model with repeated stratified k fold")
    print(f"Balanced Accuracy: {np.mean(cv_results['test_balanced_accuracy']):.3f} ± {np.std(cv_results['test_balanced_accuracy']):.3f}")
    print(f"Precision Score: {np.mean(cv_results['test_precision']):.3f} ± {np.std(cv_results['test_precision']):.3f}")
    print(f"Recall Score: {np.mean(cv_results['test_recall']):.3f} ± {np.std(cv_results['test_recall']):.3f}")
    print(f"F1 Score: {np.mean(cv_results['test_f1']):.3f} ± {np.std(cv_results['test_f1']):.3f}")
    print()
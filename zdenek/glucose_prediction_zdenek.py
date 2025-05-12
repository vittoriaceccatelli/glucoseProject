from xgboost import XGBRegressor
import pandas as pd 
import numpy as np
from sklearn.model_selection import  LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import torch
from sklearn.ensemble import RandomForestRegressor

#GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#import labeled features and labeled demographics 
data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features_zdenek_1s.csv")

#merge demographics and labeled features, drop unneccesary columns
data = data.dropna(subset=['Glucose_Mean'])

#get names of feature columns and divide data into features and labels and get group IDs (participant IDs)
feature_columns = [col for col in data.columns if col not in ['PersStatus', 'Glucose_Mean', 'Time', 'Participant_ID']]

X = data[feature_columns]

y = data['Glucose_Mean']
groups = data['Participant_ID']

#define parameters to be used for XGBRegressor
xgb_params = {
    "max_depth": 6,
    "n_estimators": 100,
    "learning_rate": 0.1}



#PERSONALIZED MODEL 
rmse_scores = []
mape_scores = []

for participant_id in groups.unique():
    #get participant data
    participant_data = data[data["Participant_ID"] == participant_id]

    #divide data into two parts, first half is training data and second half is test data
    mid_point = len(participant_data) // 2
    train_data = participant_data.iloc[:mid_point]
    test_data = participant_data.iloc[mid_point:]

    #fit random forest regressor for feature importance
    rf = RandomForestRegressor(n_estimators=50,random_state=42)
    rf.fit(train_data[feature_columns], train_data['Glucose_Mean'])

    #select features with an importance of over 0.005
    feature_importances = pd.Series(rf.feature_importances_, index=feature_columns)
    selected_features = feature_importances[feature_importances > 0.005].index.tolist()

    #get data from selected features
    X_train, y_train = train_data[selected_features], train_data['Glucose_Mean']
    X_test, y_test = test_data[selected_features], test_data['Glucose_Mean']

    #train xgb Regressor with specified parameters and fit model
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    #predict labels from testing data
    y_pred = model.predict(X_test)

    #calculate RMSE, MAPE and accuracy 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    accuracy = 100 - mape

    rmse_scores.append(rmse)
    mape_scores.append(mape)

    print(f"Participant {participant_id} - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Accuracy: {accuracy:.2f}%")

print(f"Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Mean MAPE: {np.mean(mape_scores):.2f}% ± {np.std(mape_scores):.2f}%")
print(f"Mean Accuracy: {100 - np.mean(mape_scores):.2f}% ± {np.std(mape_scores):.2f}%")



#POPULATION MODEL: leave one group out as test
logo = LeaveOneGroupOut()
rmse_scores = []
mape_scores = []

for train_idx, test_idx in logo.split(X, y, groups):
    #use random forest regressor for esitmation of feature importance and fit model
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])

    #select features if they have an importance of more than 0.005
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    selected_features = feature_importances[feature_importances > 0.005].index.tolist()

    #get data from selected features 
    X_train, X_test = X.iloc[train_idx][selected_features], X.iloc[test_idx][selected_features]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    #train XGBRegressor with defined xgb parameters
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    #predict on test data
    y_pred = model.predict(X_test)

    #calculate RMSE, MAPE and accuracy 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100 
    accuracy = 100 - mape 
    rmse_scores.append(rmse)
    mape_scores.append(mape)

    print(f"Fold {len(rmse_scores)} - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Accuracy: {accuracy:.2f}%")

print(f"Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Mean MAPE: {np.mean(mape_scores):.2f}% ± {np.std(mape_scores):.2f}%")
print(f"Mean Accuracy: {100 - np.mean(mape_scores):.2f}%")

#what i did: overlapping data, stratified k fold, weight without balancing
#i think discrepancy in classification (and feature importance) is probably due to different class balancing 

#--> NO FOOD LOG
#--> DEEP LEARNING 
#--> NO CLASS BALANCING, WEIGHTING
#--> DATA LEAKAGE 

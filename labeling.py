import pandas as pd
features_path = r'D:\Vittoria\Code\data'

#load dexcom classified data
classified_file_path = rf'{features_path}\dexcom\dexcom_classified_and_nonzero.csv'
classified_data = pd.read_csv(classified_file_path)
classified_data.rename(columns={"Timestamp": "Time"}, inplace=True)

#load feature data 
file_path = rf'{features_path}\all_features.csv'
data = pd.read_csv(file_path)
data = data[(data.iloc[:, 4] != 0)]
data = data[(data.iloc[:, 13] != 0)]
data = data[(data.iloc[:, 20] != 0)]
data.rename(columns={"Time": "Hours_from_midnight"}, inplace=True)
data.rename(columns={"Time.1": "Minutes_from_midnight"}, inplace=True)
data.rename(columns={"Unnamed: 0": "Time"}, inplace=True)

#merge dexcom and feature data on time and participant ID and save labeled data
labeled_data = pd.merge(data, classified_data, on=["Time", "Participant_ID"], how="inner")
labeled_data.to_csv(r'D:\Vittoria\Code\data\labeled_features.csv', index=False) 


#DEMO
#load data
file_path = rf'{features_path}\demographics.csv'
data = pd.read_csv(file_path)

#assign normoglycemic or hyperglycemic status
data["DiabStatus"] = "Normoglycemic"
data.loc[(data["HbA1c"] >= 5.7) & (data["HbA1c"] <= 6.4), "DiabStatus"] = "Hyperglycemic"
data.to_csv(r'data\labeled_demographics.csv', index=False)

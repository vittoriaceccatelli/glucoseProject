import pandas as pd
features_path = r'D:\Vittoria\Code\data'

#load dexcom classified data
classified_file_path = rf'{features_path}\dexcom\dexcom_classified_zdenek.csv'
classified_data = pd.read_csv(classified_file_path)
classified_data.rename(columns={"Timestamp": "Time"}, inplace=True)

#load feature data
#file_path = rf'{features_path}\all_features.csv'
file_path = rf'{features_path}\all_features_zdenek_1s.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=["Unnamed: 0", "ID", "Gender", "HbA1c"])
data.rename(columns={"index": "Time"}, inplace=True)
print(data)

#merge dexcom and feature data on time and participant ID and save labeled data
labeled_data = pd.merge(data, classified_data, on=["Time", "Participant_ID"], how="inner")
labeled_data.to_csv(r'D:\Vittoria\Code\data\labeled_features_zdenek_1s.csv', index=False) 

'''
#DEMO
#load data
file_path = rf'{features_path}\features\demographics.csv'
data = pd.read_csv(file_path)

#assign normoglycemic or hyperglycemic status
data["DiabStatus"] = "Normoglycemic"
data.loc[(data["HbA1c"] >= 5.7) & (data["HbA1c"] <= 6.4), "DiabStatus"] = "Hyperglycemic"
data.to_csv(r'data\labeled_demographics.csv', index=False)
'''
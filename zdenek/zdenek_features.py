import torch
import pandas as pd
import wearablecomp
import numpy as np
import datetime as datetime

#function to compute HRV metrics 
def compute_hrv_metrics(data_ibi):
    if len(data_ibi) <= 1:
        return pd.Series(dtype="float64")
    features = {
        "SDNN": wearablecomp.SDNN(data_ibi['Var']),
        "RMSSD": wearablecomp.RMSSD(data_ibi['Var']),
        "CVNN": data_ibi['Var'].std() / data_ibi['Var'].mean(),
        "MeanNN": data_ibi['Var'].mean() * 1000,
        "SDSD": np.std(np.diff(data_ibi['Var'])) * 1000
    }
    return pd.DataFrame([features], index=[data_ibi.index[0]])

def compute_acc_metrics(data):
    if len(data) <= 1:
        return pd.Series(dtype="float64")
    features = {
        "ACC_Median": data['Var'].median(),
        "ACC_5th": data['Var'].quantile(0.05),
        "ACC_95th": data['Var'].quantile(0.95)
    }
    return pd.DataFrame([features], index=[data.index[0]])

def compute_temp_metrics(data):
    if len(data) <= 1:
        return pd.Series(dtype="float64")
    features = {
        "TEMP_Median":  data['Var'].median(),
        "TEMP_5th": data['Var'].quantile(0.05),
        "TEMP_95th": data['Var'].quantile(0.95),
        "TEMP_Mean": data['Var'].mean(),
        "TEMP_Max": data['Var'].max(),
        "TEMP_Min": data['Var'].min(),
        "TEMP_Variability": data['Var'].std()
    }
    return pd.DataFrame([features], index=[data.index[0]])


def compute_hr_metrics(data):
    if len(data) <= 1:
        return pd.Series(dtype="float64")
    features = {
        "HR_Mean": data['Var'].mean(),
        "HR_Max": data['Var'].max(),
        "HR_Min": data['Var'].min(),
        "HR_Variability": data['Var'].std()
    }
    return pd.DataFrame([features], index=[data.index[0]])


def compute_bvp_metrics(data):
    if len(data) <= 1:
        return pd.Series(dtype="float64")
    features = {
        "BVP_PPI_Mean": data['Var'].mean(),
        "BVP_PPI_STD": data['Var'].std(),
        "BVP_Amplitude_Mean": np.mean(np.abs(data['Var'] - np.mean(data['Var'])))
    }
    return pd.DataFrame([features], index=[data.index[0]])


def compute_eda_metrics(data):
    if len(data) <= 1:
        return pd.Series(dtype="float64")
    features = {
        "EDA_Median": data['Var'].median(),
        "EDA_5th": data['Var'].quantile(0.05),
        "EDA_95th": data['Var'].quantile(0.95)
    }
    return pd.DataFrame([features], index=[data.index[0]])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#paths to raw data
scripts_dir = r'D:\Vittoria\Code'
raw_data = r'D:\Martin\For Corin\Data\raw'


#create lists to save features for all participants
all_features = []
for id in range(1,17):

    all_participants_data = pd.DataFrame()
    #ID
    formatted_id = f'{id:03}'

    #HR Features
    filepath = f'{raw_data}\{formatted_id}\HR_{formatted_id}.csv'
    data = pd.read_csv(filepath, skiprows=1, names=['Time', 'Var'])
    if formatted_id=='001':
        data['Time'] = pd.to_datetime(data["Time"], format="%m/%d/%y %H:%M", errors="coerce")
    else:
        data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    data['Var'] = pd.to_numeric(data['Var'], errors='coerce')
    data = data.dropna()
    data.set_index("Time", inplace=True)
    data.index = data.index.round('1s')
    results = data.resample("1s").apply(compute_hr_metrics)
    if 0 in results.columns:
        results.drop(0, axis=1, inplace=True)
    all_participants_data = all_participants_data.merge(results, how='outer', left_index=True, right_index=True)

    print("HR")






    #ACC Features
    filepath = f'{raw_data}\{formatted_id}\ACC_{formatted_id}.csv'
    data = pd.read_csv(filepath, names=["Time", "x", "y", "z"], dtype=str)
    data[['x', 'y', 'z']] = data[['x', 'y', 'z']].apply(pd.to_numeric, errors='coerce')
    data['Var'] = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    data = data.drop(columns=['x', 'y', 'z'])   
    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    data['Var'] = pd.to_numeric(data['Var'], errors='coerce')
    data = data.dropna()
    data.set_index("Time", inplace=True)
    data.index = data.index.round('1s')
    print(data.resample("1s"))
    results = data.resample("1s").apply(compute_acc_metrics)
    if 0 in results.columns:
        results.drop(0, axis=1, inplace=True)
    all_participants_data = all_participants_data.merge(results, how='outer', left_index=True, right_index=True)

    print("ACC")


    #HRV FEATURES: 8 features
    #Import ibi data, set Time column as index, drop NaNs
    filepath = f'{raw_data}\{formatted_id}\IBI_{formatted_id}.csv'
    data_ibi = pd.read_csv(filepath, names=['Time', 'Var'])
    data_ibi.columns = data_ibi.columns.str.strip()    
    data_ibi['Time'] = pd.to_datetime(data_ibi['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    data_ibi['Var'] = pd.to_numeric(data_ibi['Var'], errors='coerce')
    data_ibi = data_ibi.dropna()
    data_ibi.set_index("Time", inplace=True)

    #Resample ibi data every 5 minutes and apply compute_hrv_metrics function that returns pd Dataframe with 
    #HRV Max, HRV Min, HRV Mean, HRV Median, SDNN, RMSSD, NNX, pNNx
    data_ibi.index = data_ibi.index.round('1s')
    hrv_results = data_ibi.resample("1s").apply(compute_hrv_metrics)
    if 0 in hrv_results.columns:
        hrv_results.drop(0, axis=1, inplace=True)
    
    #append HRV metrics for specific participant to list all_hrv
    all_participants_data = all_participants_data.merge(hrv_results, how='outer', left_index=True, right_index=True)

    print("HRV")

    #TEMP Features
    filepath = f'{raw_data}\{formatted_id}\TEMP_{formatted_id}.csv'
    data = pd.read_csv(filepath, names=['Time', 'Var'])
    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    data['Var'] = pd.to_numeric(data['Var'], errors='coerce')
    data = data.dropna()
    data.set_index("Time", inplace=True)
    data.index = data.index.round('1s')
    results = data.resample("1s").apply(compute_temp_metrics)
    if 0 in results.columns:
        results.drop(0, axis=1, inplace=True)
    all_participants_data = all_participants_data.merge(results, how='outer', left_index=True, right_index=True)

    print("TEMP")


    #BVP Features
    filepath = f'{raw_data}\{formatted_id}\BVP_{formatted_id}.csv'
    data = pd.read_csv(filepath, names=['Time', 'Var'])
    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    data['Var'] = pd.to_numeric(data['Var'], errors='coerce')
    data = data.dropna()
    data.set_index("Time", inplace=True)
    data.index = data.index.round('1s')
    results = data.resample("1s").apply(compute_bvp_metrics)
    if 0 in results.columns:
        results.drop(0, axis=1, inplace=True)
    all_participants_data = all_participants_data.merge(results, how='outer', left_index=True, right_index=True)

    print("BVP")


    #EDA Features
    filepath = f'{raw_data}\{formatted_id}\EDA_{formatted_id}.csv'
    data = pd.read_csv(filepath, names=['Time', 'Var'])
    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    data['Var'] = pd.to_numeric(data['Var'], errors='coerce')
    data = data.dropna()
    data.set_index("Time", inplace=True)
    data.index = data.index.round('1s')
    results = data.resample("1s").apply(compute_eda_metrics)
    if 0 in results.columns:
        results.drop(0, axis=1, inplace=True)
    all_participants_data = all_participants_data.merge(results, how='outer', left_index=True, right_index=True)
    
    print("EDA")


    #ID column
    if "Participant_ID" not in all_participants_data.columns:
        all_participants_data.insert(0, "Participant_ID", formatted_id)
    else:
        all_participants_data["Participant_ID"] = formatted_id



    #substitute NaNs with 0s and add to all_features list
    all_participants_data = all_participants_data.fillna(0)
    all_participants_data.reset_index(inplace=True)  # Ensures 'Time' is kept
    all_features.append(all_participants_data)

    print(all_features)
    print(formatted_id)


all_features = pd.concat(all_features, ignore_index=True)  # Keep everything as a DataFrame, not a list

#DEMOGRAPHICS: 
filepath = f'{raw_data}\Demographics.csv'
demo = pd.read_csv(filepath, skiprows=1, names=['ID','Gender','HbA1c'])
demo['ID'] = pd.to_numeric(demo['ID'], errors='coerce')
demo['HbA1c'] = pd.to_numeric(demo['HbA1c'], errors='coerce')
demo['Gender'] = demo['Gender'].map({'MALE': 0, 'FEMALE': 1})

all_features["Participant_ID"] = all_features["Participant_ID"].astype(str)
demo["ID"] = demo["ID"].astype(str)

all_features = all_features.merge(demo, left_on='Participant_ID', right_on='ID', how='left')

#save CSV
all_features.to_csv(r'D:\Vittoria\Code\data\all_features_zdenek_1s.csv', index=True)
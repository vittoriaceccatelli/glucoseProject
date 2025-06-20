import torch
import pandas as pd
import wearablecomp
import food_log
import numpy as np
import datetime as datetime
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d
from scipy import signal
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

#function to compute HRV metrics 
def compute_hrv_metrics(data_ibi):
    if len(data_ibi) <= 1:
        return pd.Series(dtype="float64")
    features = {
        "HRV_Max": wearablecomp.HRV(data_ibi['Var'])[0],
        "HRV_Min": wearablecomp.HRV(data_ibi['Var'])[1],
        "HRV_Mean": wearablecomp.HRV(data_ibi['Var'])[2],
        "HRV_Median": wearablecomp.HRV(data_ibi['Var'])[3],
        "SDNN": wearablecomp.SDNN(data_ibi['Var']),
        "RMSSD": wearablecomp.RMSSD(data_ibi['Var']),
        "NNX": wearablecomp.NNx(data_ibi['Var'])[0],
        "PNNX": wearablecomp.NNx(data_ibi['Var'])[1]
    }
    return pd.DataFrame([features], index=[data_ibi.index[0]])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#paths to raw data
scripts_dir = r'D:\Vittoria\Code'
raw_data = r'D:\Martin\For Corin\Data\raw'

#Sensor data, no BVP data used in paper so excluded here as well
sensor_types = ['EDA', 'HR', 'TEMP', 'ACC']

#create lists to save features for all participants
all_features = []
for id in range(1,17):
    all_participants_data = pd.DataFrame()
    #ID
    formatted_id = f'{id:03}'

    #IBI FEATURES: 8 features
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
    data_ibi.index = data_ibi.index.round('5min')
    hrv_results = data_ibi.resample("5min").apply(compute_hrv_metrics)
    hrv_results.drop(0, axis=1, inplace=True)
    
    #append HRV metrics for specific participant to list all_hrv
    all_participants_data = pd.merge(all_participants_data, hrv_results, how='outer', left_index=True, right_index=True)




    #'EDA', 'HR', 'ACC', 'TEMP' FEATURES: 7*4+2+3 --> 33 features
    #create a list to save features from all four sensors and loop through all sensors 
    min5_features = []
    for sensor in sensor_types:
        file_name=f'{raw_data}\{formatted_id}\{sensor}_{formatted_id}.csv'
        #if sensor is ACC --> return both 5min aggregated data and 2hr aggregated data
        #otherwise return only 5min aggregated data
        if sensor=="ACC":
            data, data_2hrs = wearablecomp.e4import(file_name, sensor, formatted_id)
        else:
            data = wearablecomp.e4import(file_name, sensor, formatted_id)
            #if sensor is EDA, also  return peaks and 2hr aggregated data 
            if sensor=="EDA":
                eda = pd.read_csv(file_name, skiprows=1, names=['Time', 'Var'])
                eda_peaks = wearablecomp.PeaksEDA(eda['Var'], eda['Time'])

                #append EDA data to list for all participants 
                all_participants_data = pd.merge(all_participants_data, eda_peaks, how='outer', left_index=True, right_index=True)

        #append sensor data to list for all sensors 
        min5_features.append(data)
    
    #append sensor data to list for all participants
    participant_df = pd.concat(min5_features, axis=1)
    participant_df = participant_df.loc[:, ~participant_df.columns.duplicated()]
    participant_df['ind'] = pd.to_datetime(participant_df['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    participant_df.set_index("ind", inplace=True)
    participant_df = participant_df.drop(columns=[col for col in participant_df.columns if col.startswith("Time")])
    all_participants_data = pd.merge(all_participants_data, participant_df, how='outer', left_index=True, right_index=True)




    #ACTIVITY BOUTS FEATURES --> 3 features
    #import heart rate and acc data and calculate corresponding features
    file_name_acc=f'{raw_data}\{formatted_id}\ACC_{formatted_id}.csv'
    file_name_hr=f'{raw_data}\{formatted_id}\HR_{formatted_id}.csv'
    data_acc, data_acc2hrs = wearablecomp.e4import(file_name_acc, "ACC", formatted_id)
    data_hr = wearablecomp.e4import(file_name_hr, "HR", formatted_id)
    time=data_hr['Time']
    #use hr and acc data to calculate activity bouts, set time as index and calculate activity every 24h and every hour
    activity_bouts = wearablecomp.exercisepts(data_acc, data_hr, time)
    activity_bouts.set_index("Time", inplace=True)
    activity_bouts['Activity Bouts_24hr'] = activity_bouts['Activity Bouts'].rolling("24h").sum()/24
    activity_bouts['Activity Bouts_1hr'] = activity_bouts['Activity Bouts'].rolling("1h").sum()




    #ACC 2 hour FEATURES --> 2 features
    all_participants_data = pd.merge(all_participants_data, data_acc2hrs, how='outer', left_index=True, right_index=True)




    #WAKE TIME FEATURES --> 1 feature
    #use acc and hr data to calculate wake_time
    wake_time = wearablecomp.wake_time(data_acc, data_hr)

    #merge wake_time and activity bouts
    wake_and_act = activity_bouts.merge(wake_time, left_index=True, right_index=True, how='inner')

    #append to lists for all participants 
    all_participants_data = pd.merge(all_participants_data, wake_and_act, how='outer', left_index=True, right_index=True)




    #CIRCADIAN RHYTHM --> 2 features 
    #get time column from hr data
    file_name_hr=f'{raw_data}\{formatted_id}\HR_{formatted_id}.csv'
    data_hr = wearablecomp.e4import(file_name_hr, "HR", formatted_id)

    #calculate hours and minutes from midnight
    hourfrommid = data_hr['Time'].dt.hour
    minfrommid = data_hr['Time'].dt.hour * 60 + data_hr['Time'].dt.minute

    #concatenate to one dataframe 
    circrhythm = pd.concat([hourfrommid, minfrommid], axis=1)

    #time column for circadian rhythm, concatenate with circadian features
    start_time = data_hr['Time'].min()
    end_time = data_hr['Time'].max()
    timestamps = pd.date_range(start=start_time, end=end_time, freq='5min')
    timestamp_series = pd.Series(timestamps, name="date")
    circrhythm = pd.concat([timestamp_series, circrhythm], axis=1)
    circrhythm = circrhythm.set_index("date")
    all_participants_data = pd.merge(all_participants_data, circrhythm, how='outer', left_index=True, right_index=True)




    #FOOD LOG FEATURES --> 19 features
    #get data
    filepath = f'{raw_data}\{formatted_id}\Food_Log_{formatted_id}.csv'
    data_food_log = pd.read_csv(filepath, skiprows=1, names=['date','time','time_begin','time_end','logged_food','amount','unit','searched_food','calorie','total_carb','dietary_fiber','sugar','protein','total_fat'])
    
    #create time_begin column
    data_food_log['time_begin'] = pd.to_datetime(data_food_log['time_begin'])
    
    #calculate food log features
    food_2hrs = food_log.food_log_features('2h', data_food_log)
    food_8hrs = food_log.food_log_features('8h', data_food_log)
    food_24hrs = food_log.food_log_features('24h', data_food_log)

    #set time as index and add to participant data
    food_2hrs = food_2hrs.set_index("start_time")
    all_participants_data = pd.merge(all_participants_data, food_2hrs, how='outer', left_index=True, right_index=True)
    food_8hrs = food_8hrs.set_index("start_time")
    all_participants_data = pd.merge(all_participants_data, food_8hrs, how='outer', left_index=True, right_index=True)
    food_24hrs = food_24hrs.set_index("start_time")
    all_participants_data = pd.merge(all_participants_data, food_24hrs, how='outer', left_index=True, right_index=True)
    



    #ID column
    if "Participant_ID" not in all_participants_data.columns:
        all_participants_data.insert(0, "Participant_ID", formatted_id)
    else:
        all_participants_data["Participant_ID"] = formatted_id

    #substitute NaNs with 0s and add to all_features list
    all_participants_data = all_participants_data.fillna(0)
    all_features.append(all_participants_data)
    print(all_features)
    print(formatted_id)

#save CSV
all_features = pd.concat(all_features)
all_features.to_csv(r'D:\Vittoria\Code\data\other\all_features.csv', index=True)

#DEMOGRAPHICS: 
filepath = f'{raw_data}\Demographics.csv'
demo = pd.read_csv(filepath, skiprows=1, names=['ID','Gender','HbA1c'])
demo['ID'] = pd.to_numeric(demo['ID'], errors='coerce')
demo['HbA1c'] = pd.to_numeric(demo['HbA1c'], errors='coerce')
demo = demo.set_index(demo['ID']).sort_index()
demo.to_csv('data\demographics.csv', index=False)

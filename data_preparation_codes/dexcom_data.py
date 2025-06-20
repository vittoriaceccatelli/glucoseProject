import torch
import pandas as pd
import datetime as datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scripts_dir = r'D:\Vittoria\Code'
raw_data = r'D:\Martin\For Corin\Data\raw'
raw_zipped = r'C:/Users/vceccatelli/.vscode/Code/data/raw.zip'

resampled_list = []

#loop through participants and load data
for id in range(1, 17):
    formatted_id = f"{id:03}"
    filepath = f'{raw_data}\{formatted_id}\Dexcom_{formatted_id}.csv'

    #read Dexcom csv file for participant
    data_ibi = pd.read_csv(filepath, skiprows=13, 
        names=['Index', 'Timestamp', 'Event Type', 'Event Subtype', 'Patient Info',
                'Device Info', 'Source Device ID', 'Glucose Value', 'Insulin Value',
                'Carb Value', 'Duration', 'Glucose Rate of Change', 'Transmitter Time'], 
        parse_dates=["Timestamp"])

    #drop unnecessary/empty columns
    to_drop = ['Index', 'Event Type', 'Transmitter Time', 'Event Subtype', 
                'Source Device ID', 'Patient Info', 'Device Info', 
                'Insulin Value', 'Carb Value', 'Duration', 'Glucose Rate of Change']
    data_ibi.drop(columns=[col for col in to_drop if col in data_ibi.columns], inplace=True)

    #set index
    data_ibi.set_index("Timestamp", inplace=True)

    #resample dexcom data every 5mins (actually it is already sampled every 5 mins but round it down to lower 5min stamp)
    resampled = data_ibi.resample("5T").agg({
        "Glucose Value": ["mean"]
    })
    resampled.columns = ["Glucose_Mean"]

    #reset index and insert column specifiying participant
    resampled.reset_index(inplace=True)
    resampled["Participant_ID"] = formatted_id
    resampled_list.append(resampled)

#concatenate all participants and save csv
final_resampled_df = pd.concat(resampled_list)
final_resampled_df.to_csv(f"D:\Vittoria\Code\data\dexcom\dexcom_data_5mins.csv", index=False)
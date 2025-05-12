import pandas as pd

def food_log_features(window, data_food_log):
    #create 5min intervals for dataframe
    min_time = data_food_log["time_begin"].min().floor('5min')
    max_time = data_food_log["time_begin"].max().ceil('5min')
    all_intervals = pd.date_range(start=min_time, end=max_time, freq="5min")
    full_time_df = pd.DataFrame({"start_time": all_intervals})
    
    #for every interval create a window sized window
    overlapping_windows = []
    for interval_start in all_intervals:
        window_end = interval_start + pd.Timedelta(window)
        window_data = data_food_log[(data_food_log["time_begin"] >= interval_start) & (data_food_log["time_begin"] < window_end)]
        #calculate features 
        if not window_data.empty:
            aggregated_data = {
                f"start_time": interval_start,
                f"Calories_{window}": window_data["calorie"].sum(),
                f"Protein_{window}": window_data["protein"].sum(),
                f"Sugar_{window}": window_data["sugar"].sum(),
                f"Carbs_{window}": window_data["total_carb"].sum(),
                f"Eat_cnt_{window}": window_data["calorie"].count()
            }
            overlapping_windows.append(aggregated_data)
    
    result_df = pd.DataFrame(overlapping_windows)
    
    # Merge with full_time_df to ensure all intervals are included
    result_df = full_time_df.merge(result_df, on="start_time", how="left").fillna(0)
    
    # calculate additional features 
    result_df[f"Eat_mean_{window}"] = result_df[f"Calories_{window}"] / result_df[f"Eat_cnt_{window}"].replace(0, 1)
    if window == '2h':
        result_df["Eat"] = (result_df[f"Eat_cnt_{window}"] > 0).astype(int)
    
    return result_df
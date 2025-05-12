import pandas as pd
import matplotlib.pyplot as plt


pred_df = pd.read_csv("detailed_predictions.csv")
full_data = pd.read_csv("D:\\Vittoria\\Code\\data\\dexcom\\dexcom_data_5mins.csv")
full_data["Timestamp"] = pd.to_datetime(full_data["Timestamp"])
full_data["time_idx"] = ((full_data["Timestamp"] - full_data["Timestamp"].min()).dt.total_seconds() // 60).astype(int)

for participant_id in full_data["Participant_ID"].unique():
    full_series = full_data[full_data["Participant_ID"] == participant_id].copy()
    preds = pred_df[pred_df["Participant_ID"] == participant_id].copy()

    full_series = full_series.sort_values("time_idx")
    preds = preds.sort_values("time_idx")

    last_time_idx = full_series["time_idx"].max() - len(preds)

    pred_time_idx = list(range(last_time_idx + 1, last_time_idx + 1 + len(preds)))

    plt.figure(figsize=(12, 5))
    plt.plot(full_series["time_idx"], full_series["Glucose_Mean"], label="Historical Target", color="gray")
    plt.plot(pred_time_idx, preds["Predicted"], label="Predicted", linestyle="-", color="red")
    plt.title(f"Participant {participant_id} — Full Series with Future Forecast")
    plt.xlabel("Sequential Time Step")
    plt.ylabel("Glucose")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"participant_{participant_id}_forecast_plot.png")
    plt.close()

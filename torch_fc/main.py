import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import MAPE, RMSE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from pytorch_forecasting.data.encoders import NaNLabelEncoder

# Set seed for reproducibility
pl.seed_everything(42)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load dataset and demographics
data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\demographics.csv")
data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")

# Keep only necessary columns (features + target)
columns_to_keep = ["Time", "Participant_ID", "HRV_Max", "HRV_Min", "HRV_Mean", "HRV_Median", "SDNN", "RMSSD", "NNX", "PNNX", 
                   "Peak_eda_5min_sum", "Peak_eda_2hr_sum", "Peak_eda_2hr_mean", "EDA_Mean", "EDA_Std", 
                   "EDA_Min", "EDA_Max", "EDA_q1G", "EDA_q3G", "EDA_skew", "HR_Mean", "HR_Std", "HR_Min", 
                   "HR_Max", "HR_q1G", "HR_q3G", "HR_skew", "TEMP_Mean", "TEMP_Std", "TEMP_Min", "TEMP_Max", 
                   "TEMP_q1G", "TEMP_q3G", "TEMP_skew", "ACC_Mean", "ACC_Std", "ACC_Min", "ACC_Max", 
                   "ACC_q1G", "ACC_q3G", "ACC_skew", "ACC_Mean2hrs", "ACC_Max2hrs", 
                   "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr","Wake_Time","Hours_from_midnight",
                   "Minutes_from_midnight", "Gender", "HbA1c", "Calories_2h", "Protein_2h", "Sugar_2h",
                   "Carbs_2h","Eat_cnt_2h","Eat_mean_2h","Eat","Calories_8h","Protein_8h","Sugar_8h","Carbs_8h",
                   "Eat_cnt_8h","Eat_mean_8h","Calories_24h","Protein_24h","Sugar_24h","Carbs_24h","Eat_cnt_24h",
                   "Eat_mean_24h", "Glucose_Mean"]

data = data[columns_to_keep]
data = data.rename(columns={"Glucose_Mean": "target"})

# Create time index per participant based on time in minutes from participant start
data["time_idx"] = (pd.to_datetime(data["Time"]) - pd.to_datetime(data["Time"]).min()).dt.total_seconds() // 60
data["time_idx"] = data.groupby("Participant_ID")["time_idx"].transform(lambda x: x - x.min())
data["time_idx"] = pd.to_numeric(data["time_idx"], downcast='integer')

# Convert Participant_ID to string for categorical handling
data["Participant_ID"] = data["Participant_ID"].astype(str)

# Drop original timestamp
data = data.drop(columns=["Time"])

# Set sequence lengths
max_prediction_length = 1
max_encoder_length = 288
training_cutoff = data["time_idx"].max() - max_prediction_length

# Initialize performance metric lists
mape_scores = []
rmse_scores = []
participant_ids = data["Participant_ID"].unique()


if __name__ == '__main__':
    # Define training TimeSeriesDataSet
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["Participant_ID"],
        min_encoder_length=max_encoder_length//2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Gender", "Participant_ID"],
        static_reals=["HbA1c"],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx","Hours_from_midnight", "Minutes_from_midnight", "HRV_Max", "HRV_Min", "HRV_Mean", "HRV_Median", "SDNN", "RMSSD", "NNX", "PNNX", 
                    "Peak_eda_5min_sum", "Peak_eda_2hr_sum", "Peak_eda_2hr_mean", "EDA_Mean", "EDA_Std", 
                    "EDA_Min", "EDA_Max", "EDA_q1G", "EDA_q3G", "EDA_skew", "HR_Mean", "HR_Std", "HR_Min", 
                    "HR_Max", "HR_q1G", "HR_q3G", "HR_skew", "TEMP_Mean", "TEMP_Std", "TEMP_Min", "TEMP_Max", 
                    "TEMP_q1G", "TEMP_q3G", "TEMP_skew", "ACC_Mean", "ACC_Std", "ACC_Min", "ACC_Max", 
                    "ACC_q1G", "ACC_q3G", "ACC_skew", "ACC_Mean2hrs", "ACC_Max2hrs", 
                    "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr","Wake_Time", "Calories_2h", "Protein_2h", "Sugar_2h",
                    "Carbs_2h","Eat_cnt_2h","Eat_mean_2h","Eat","Calories_8h","Protein_8h","Sugar_8h","Carbs_8h",
                    "Eat_cnt_8h","Eat_mean_8h","Calories_24h","Protein_24h","Sugar_24h","Carbs_24h","Eat_cnt_24h", "Eat_mean_24h"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["Participant_ID"]),
        categorical_encoders={"Participant_ID": NaNLabelEncoder().fit(participant_ids)},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    # Build validation set from test participant
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)


    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
    val_dataloader = validation.to_dataloader(train=False, batch_size=16)

    # Callbacks and logging
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=5, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=25,
        devices=1,
        strategy="auto",
        enable_model_summary=True,
        gradient_clip_val=0.06,
        limit_train_batches=50,
        fast_dev_run=False,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger)

    # Initialize Temporal Fusion Transformer
    best_tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=60,
        attention_head_size=7,
        dropout=0.2,
        hidden_continuous_size=19,
        loss=QuantileLoss(),
        log_interval=-1,
        optimizer="adam",
        reduce_on_plateau_patience=3)

    print(f"Number of parameters in network: {best_tft.size() / 1e3:.1f}k")

    # Train the model
    trainer.fit(best_tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Make raw predictions (for optional plotting)
    prediction_output, x, y, a, c = best_tft.predict(val_dataloader, mode="raw", return_x=True, return_y=True, trainer_kwargs=dict(accelerator="cpu"))

    # Get predicted values
    prediction = best_tft.predict(val_dataloader, return_x=True, return_y=True, trainer_kwargs=dict(accelerator="cpu"))

    # Calculate performance metrics
    mape = MAPE()(prediction.output, prediction.y).item()
    rmse = RMSE()(prediction.output, prediction.y).item()

    mape_scores.append(mape)
    rmse_scores.append(rmse)

# Print aggregated results across all participants
mape_scores = np.array(mape_scores)
rmse_scores = np.array(rmse_scores)

print(f"MAPE: {mape_scores.mean():.5f} ± {mape_scores.std():.5f}")
print(f"RMSE: {rmse_scores.mean():.5f} ± {rmse_scores.std():.5f}")

# Optional: plot prediction samples
for idx in range(len(prediction_output[0].detach().cpu().numpy())):
    fig = best_tft.plot_prediction(x, prediction_output, idx=idx, add_loss_to_title=True)
    # fig.savefig(rf"D:\Vittoria\Code\torch_fc\other\forecast_sample_{idx+1}.png", dpi=300, bbox_inches="tight")
    # plt.close(fig)
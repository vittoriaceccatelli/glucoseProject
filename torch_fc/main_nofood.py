from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import MAPE, RMSE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (optimize_hyperparameters,)
import matplotlib.pyplot as plt
            
pl.seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#PREDICTION
data = pd.read_csv(r"D:\Vittoria\Code\data\labeled_features.csv")
demo = pd.read_csv(r"D:\Vittoria\Code\data\demographics.csv")
data = pd.merge(data, demo, left_on="Participant_ID", right_on="ID")

columns_to_keep = ["Time", "Participant_ID", "HRV_Max", "HRV_Min", "HRV_Mean", "HRV_Median", "SDNN", "RMSSD", "NNX", "PNNX", 
                   "Peak_eda_5min_sum", "Peak_eda_2hr_sum", "Peak_eda_2hr_mean", "EDA_Mean", "EDA_Std", 
                   "EDA_Min", "EDA_Max", "EDA_q1G", "EDA_q3G", "EDA_skew", "HR_Mean", "HR_Std", "HR_Min", 
                   "HR_Max", "HR_q1G", "HR_q3G", "HR_skew", "TEMP_Mean", "TEMP_Std", "TEMP_Min", "TEMP_Max", 
                   "TEMP_q1G", "TEMP_q3G", "TEMP_skew", "ACC_Mean", "ACC_Std", "ACC_Min", "ACC_Max", 
                   "ACC_q1G", "ACC_q3G", "ACC_skew", "ACC_Mean2hrs", "ACC_Max2hrs", 
                   "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr","Wake_Time","Hours_from_midnight",
                   "Minutes_from_midnight", "Gender", "HbA1c", "Glucose_Mean"]

data = data[columns_to_keep]
data = data.rename(columns={"Glucose_Mean": "target"})

data["time_idx"] = (pd.to_datetime(data["Time"]) - pd.to_datetime(data["Time"]).min()).dt.total_seconds() // 60
data["time_idx"] = data.groupby("Participant_ID")["time_idx"].transform(lambda x: x - x.min())
data["time_idx"] = pd.to_numeric(data["time_idx"], downcast='integer')


data["Participant_ID"] = data["Participant_ID"].astype(str)

data = data.drop(columns=["Time"])

max_prediction_length = 288
min_prediction_length= 1
max_encoder_length = 2880
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="target",
    group_ids=["Participant_ID"],
    min_encoder_length=max_encoder_length//2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=min_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["Gender", "Participant_ID"],
    static_reals=["HbA1c"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx", "HRV_Max", "HRV_Min", "HRV_Mean", "HRV_Median", "SDNN", "RMSSD", "NNX", "PNNX", 
                   "Peak_eda_5min_sum", "Peak_eda_2hr_sum", "Peak_eda_2hr_mean", "EDA_Mean", "EDA_Std", 
                   "EDA_Min", "EDA_Max", "EDA_q1G", "EDA_q3G", "EDA_skew", "HR_Mean", "HR_Std", "HR_Min", 
                   "HR_Max", "HR_q1G", "HR_q3G", "HR_skew", "TEMP_Mean", "TEMP_Std", "TEMP_Min", "TEMP_Max", 
                   "TEMP_q1G", "TEMP_q3G", "TEMP_skew", "ACC_Mean", "ACC_Std", "ACC_Min", "ACC_Max", 
                   "ACC_q1G", "ACC_q3G", "ACC_skew", "ACC_Mean2hrs", "ACC_Max2hrs", 
                   "Activity Bouts","Activity Bouts_24hr","Activity Bouts_1hr","Wake_Time","Hours_from_midnight",
                   "Minutes_from_midnight"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "target"
    ],
    target_normalizer=EncoderNormalizer(),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

batch_size = 32

train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10)

if __name__ == '__main__':    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=5, verbose=False, mode="min")

    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        accelerator="cuda",
        max_epochs=25,
        devices=1,
        strategy="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,
        fast_dev_run=False,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger)

    best_tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=25,
        attention_head_size=7,
        dropout=0.3,
        hidden_continuous_size=15,
        loss=MAPE(),
        log_interval=-1,
        optimizer="adam",
        reduce_on_plateau_patience=3)

    print(f"Number of parameters in network: {best_tft.size() / 1e3:.1f}k")

    trainer.fit(best_tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    prediction_output, x, y, a, c = best_tft.predict(val_dataloader, mode = "raw", return_x=True, return_y=True, trainer_kwargs=dict(accelerator="cpu"))

    records = []

    for i in range(len(prediction_output)):
        participant_id = x["groups"][i, 0].item() 
        time_indices = x["decoder_time_idx"][i].detach().cpu().numpy()
        preds = prediction_output[i].detach().cpu().numpy() 

        if len(time_indices) != len(preds):
            print(f"Mismatch in length between time indices and predictions: {len(time_indices)} vs {len(preds)}")
            continue

        for j, time_idx in enumerate(time_indices):
            if j < len(preds):
                pred_value = preds[j] if preds.ndim == 1 else preds[j][0]
                records.append({
                    "Participant_ID": participant_id,
                    "time_idx": time_idx,
                    "Predicted": pred_value})

    df = pd.DataFrame(records)
    df.to_csv(rf"D:\Vittoria\Code\torch_fc\detailed_predictions_nofood.csv", index=False)

    for idx in range(len(preds)):
        fig = best_tft.plot_prediction(x, prediction_output, idx=idx, add_loss_to_title=True)
        fig.savefig(rf"D:\Vittoria\Code\torch_fc\plots\forecast_sample_{idx+1}_nofood.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    prediction = best_tft.predict(val_dataloader, return_x=True, return_y=True, trainer_kwargs=dict(accelerator="cpu"))

    print(MAPE()(prediction.output, prediction.y))
    print(RMSE()(prediction.output, prediction.y))

    prediction, x, _, _, _ = best_tft.predict(val_dataloader, return_x=True, trainer_kwargs=dict(accelerator="cpu"))
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, prediction)
    figures_dict = best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    for variable_name, fig in figures_dict.items():
        fig.savefig(rf"D:\Vittoria\Code\torch_fc\plots\prediction_vs_actual_{variable_name}_nofood.png", bbox_inches="tight")
        plt.close(fig)

    raw_predictions = best_tft.predict(val_dataloader, mode = "raw", trainer_kwargs=dict(accelerator="cpu"))
    interpretation = best_tft.interpret_output(raw_predictions, reduction="mean")
    best_tft.plot_interpretation(interpretation)
    plt.savefig(rf"D:\Vittoria\Code\torch_fc\plots\tft_interpretation_nofood.png")
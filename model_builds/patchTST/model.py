import torch
from pytorch_lightning import Trainer
import os
import pandas as pd
import warnings
import time  # Import the time module

from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.patch_tst import PatchTSTEstimator

warnings.filterwarnings("ignore")

# Load and preprocess data
file_path = "/Users/charlesmiller/Documents/Code/gluonts/model_builds/datasets/full.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)
df['v'] = df.index
df.reset_index(inplace=True, drop=True)
df = df.loc[df['symbol'].isin(['AAPL','SPY','QQQ','AMD','NVDA'])]
print(len(df))
df['time_idx'] = df.groupby('symbol').cumcount()
feature_columns = ['o','v','c','h','l','vw']
feature_df = df[['o','v','c','h','l','vw','time_idx','symbol']]
feature_df[feature_columns] = feature_df[feature_columns].astype('float32')

# Split data into training, validation, and test sets
train_df = feature_df[feature_df['time_idx'] < 80000]
validation_df = feature_df.loc[(feature_df['time_idx'] >= 80000) & (feature_df['time_idx'] < 90000)]
test_df = feature_df[feature_df['time_idx'] >= 90000]

print(f"Train shape: {train_df.shape}")
print(f"Train Symbols: {train_df['symbol'].unique()}")
print(f"Validation shape: {validation_df.shape}")
print(f"Validation Symbols: {validation_df['symbol'].unique()}")
print(f"Test shape: {test_df.shape}")
print(f"Test Symbols: {test_df['symbol'].unique()}")

# Create GluonTS datasets
training_data = PandasDataset.from_long_dataframe(
    train_df, timestamp="time_idx", target="c", item_id="symbol")
validation_data = PandasDataset.from_long_dataframe(
    validation_df, timestamp="time_idx", target="c", item_id="symbol")
testing_data = PandasDataset.from_long_dataframe(
    test_df, timestamp="time_idx", target="c", item_id="symbol")

# grouper = MultivariateGrouper(max_target_dim=1)
# training_data = grouper(training_data)
# validation_data = grouper(validation_data)
# testing_data = grouper(testing_data)

# Set device to CPU
device = torch.device("mps")

# Define the estimator
estimator = PatchTSTEstimator(
    # freq='15min',  # Set frequency to 15 minutes
    prediction_length=8,
    patch_len=2,
    context_length=96,
    batch_size=256,
    num_batches_per_epoch=512,
    dropout=0.2,
    )

# Set device to CPU explicitly in PyTorch Lightning Trainer
trainer = Trainer(
    max_epochs=50,  # Adjust as needed
    accelerator='gpu',  # Use 'cpu' or 'gpu' based on your preference
    # num_workers=4,  # Set to 0 to avoid issues with DataLoader
)

# Track training time
start_time = time.time()  # Record the start time

# Train the model
predictor = estimator.train(
    training_data=training_data, validation_data=validation_data, trainer=trainer)

# Record the end time for training
training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Track prediction time
start_time = time.time()  # Record the start time for prediction

# Make predictions
predictions = predictor.predict(testing_data)

# Record the end time for prediction
prediction_time = time.time() - start_time
print(f"Prediction Time: {prediction_time:.2f} seconds")

# Iterate over predictions and print them
for forecast in predictions:
    print(forecast.mean)

# Total time
total_time = training_time + prediction_time
print(f"Total Time: {total_time:.2f} seconds")

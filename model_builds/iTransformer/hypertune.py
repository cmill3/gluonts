import optuna
import torch
import time
import pandas as pd
from gluonts.dataset.split import split
from gluonts.evaluation import Evaluator
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.i_transformer import ITransformerEstimator
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

class ITransformerTuningObjective:
    def __init__(
        self, prediction_length, train, validate, metric_type="mean_wQuantileLoss"
    ):
        self.prediction_length = prediction_length
        self.metric_type = metric_type

        self.train = train
        self.validate = validate

    def get_params(self, trial) -> dict:
        return {
            "nhead": trial.suggest_int("nhead", 2, 8),
            "d_model": trial.suggest_int("d_model", 15, 80),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dim_feedforward": trial.suggest_int("dim_feedforward", 20, 200),
            "context_length": trial.suggest_int("context_length", 60, 140),
            "batch_size": trial.suggest_int("batch_size", 32, 512),
            "num_batches_per_epoch": trial.suggest_int("num_batches_per_epoch", 50, 500),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        }

    def __call__(self, trial):
        params = self.get_params(trial)
        estimator = ITransformerEstimator(
            dim_feedforward=params["dim_feedforward"],
            d_model=params["d_model"]*params["nhead"],
            nhead=params["nhead"],
            prediction_length=self.prediction_length,
            context_length=params["context_length"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            num_batches_per_epoch=params["num_batches_per_epoch"],
            dropout=params["dropout"],
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "max_epochs": 3,
            },
        )
        training_data = PandasDataset.from_long_dataframe(
            self.train, timestamp="time_idx", target="c", item_id="symbol")
        validation_data = PandasDataset.from_long_dataframe(
            self.validate, timestamp="time_idx", target="c", item_id="symbol")
        validation_label = self.validate.set_index(["symbol", "time_idx"])["c"]

        grouper = MultivariateGrouper(max_target_dim=6)
        training_data = grouper(training_data)
        validation_data = grouper(validation_data)

        predictor = estimator.train(training_data, cache_data=True)
        forecast_it = predictor.predict(validation_data, num_samples=100)
        print("FORECAST IT")
        print(forecast_it)
        print("labels")
        print(validation_label)

        forecasts = list(forecast_it)
        # Convert the forecasts to the structure expected by the evaluator
        forecasts = [forecast.mean for forecast in forecasts]
        print("FORECASTS")
        print(forecasts)

        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        print("FORECAST LEN")
        print(len(forecasts))
        agg_metrics, item_metrics = evaluator(
            validation_label, forecasts, num_series=len(forecasts)
        )
        return agg_metrics[self.metric_type]
    

# Load and preprocess data
if __name__ == "__main__":
    file_path = "/Users/charlesmiller/Documents/Code/gluonts/model_builds/datasets/full.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df['v'] = df.index
    df.reset_index(inplace=True, drop=True)
    df = df.loc[df['symbol'].isin(['AAPL','SPY','QQQ','AMD','NVDA'])]
    print(len(df))
    df['time_idx'] = df.groupby('symbol').cumcount()
    feature_df = df[['o','v','c','h','l','vw','time_idx','symbol']]

    # Split data into training, validation, and test sets
    train_df = feature_df[feature_df['time_idx'] < 80000]
    validation_df = feature_df.loc[(feature_df['time_idx'] >= 80000) & (feature_df['time_idx'] < 90000)]
    test_df = feature_df[feature_df['time_idx'] >= 90000]
    start_time = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(
        ITransformerTuningObjective(
            prediction_length=8, train=train_df, validate=validation_df
        ),
        n_trials=5,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print(time.time() - start_time)

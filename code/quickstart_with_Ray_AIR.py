import ray
from ray.air.config import ScalingConfig
from ray.data.preprocessors import MinMaxScaler
from ray.train.xgboost import XGBoostTrainer
import pandas as pd

ray.init(address="auto")
print("Connected:", ray.cluster_resources())

# --------------------------------------------
# Load dataset
# --------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
dataset = ray.data.read_csv(DATA_URL)

def create_label(df: pd.DataFrame):
    df["is_high_tip"] = (df["tip"] > 3).astype(int)
    return df

dataset = dataset.map_batches(create_label)
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3, seed=42)

# --------------------------------------------
# Preprocessing (New Ray API)
# --------------------------------------------
preprocessor = MinMaxScaler(columns=["total_bill", "tip", "size"])

# Fit on train only
preprocessor.fit(train_dataset)

# Transform
train_dataset = preprocessor.transform(train_dataset)
valid_dataset = preprocessor.transform(valid_dataset)

# --------------------------------------------
# Training
# --------------------------------------------
trainer = XGBoostTrainer(
    label_column="is_high_tip",
    num_boost_round=100,
    scaling_config=ScalingConfig(
        num_workers=1,
        use_gpu=False,
    ),
    params={
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "tree_method": "approx",
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
)

result = trainer.fit()

print("train acc = ", 1 - result.metrics["train-error"])
print("valid acc = ", 1 - result.metrics["valid-error"])
print("iterations = ", result.metrics["training_iteration"])

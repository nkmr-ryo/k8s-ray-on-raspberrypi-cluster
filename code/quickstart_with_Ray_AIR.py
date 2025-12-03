import ray
from ray.air.config import ScalingConfig
from ray.data.preprocessors import MinMaxScaler
from ray.train.xgboost import XGBoostTrainer
import pandas as pd

# --------------------------------------------
# 1. Connect to RayCluster from RayJob
# --------------------------------------------
ray.init(address="auto")
print("Connected to Ray cluster:", ray.cluster_resources())

# --------------------------------------------
# 2. Load dataset from GitHub (Seaborn tips)
# --------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"

dataset = ray.data.read_csv(DATA_URL)
print("Loaded dataset schema:", dataset.schema())

# --------------------------------------------
# 3. Create binary label for classification
#    tip > 3 → 1 (high tip)
# --------------------------------------------
def create_label(df: pd.DataFrame):
    df["is_high_tip"] = (df["tip"] > 3).astype(int)
    return df

dataset = dataset.map_batches(create_label)
print("Added label column. New schema:", dataset.schema())

# --------------------------------------------
# 4. Split train / validation
# --------------------------------------------
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3, seed=42)

# Repartition for Ray Data parallelism
train_dataset = train_dataset.repartition(num_blocks=4)
valid_dataset = valid_dataset.repartition(num_blocks=4)

# --------------------------------------------
# 5. Preprocessor (normalize numeric columns)
# --------------------------------------------
preprocessor = MinMaxScaler(
    columns=["total_bill", "tip", "size"]
)

# --------------------------------------------
# 6. Define Trainer
# --------------------------------------------
trainer = XGBoostTrainer(
    label_column="is_high_tip",
    num_boost_round=100,
    scaling_config=ScalingConfig(
        num_workers=1,     # Pi クラスタの小規模構成に最適
        use_gpu=False,
    ),
    params={
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "tree_method": "approx",
    },
    datasets={"train": train_dataset, "valid": valid_dataset},
    preprocessor=preprocessor,
)

# --------------------------------------------
# 7. Training
# --------------------------------------------
print("Starting training...")
result = trainer.fit()
print("Training completed!")

# --------------------------------------------
# 8. Report results
# --------------------------------------------
train_acc = 1 - result.metrics["train-error"]
valid_acc = 1 - result.metrics["valid-error"]
print(f"train accuracy = {train_acc:.4f}")
print(f"valid accuracy = {valid_acc:.4f}")
print(f"iterations = {result.metrics['training_iteration']}")

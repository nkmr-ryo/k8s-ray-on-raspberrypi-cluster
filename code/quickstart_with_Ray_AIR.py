import ray
from ray.air import session
from ray.air.config import ScalingConfig
from ray.data.preprocessors import MinMaxScaler
from ray.train.xgboost import XGBoostTrainer
import xgboost as xgb
import numpy as np

#####################
# 1. Ray 初期化
#####################
ray.init()  # KubeRay の RayCluster に接続される

#####################
# 2. データ読み込み & 前処理
#####################

# Seaborn の tips データセット相当を CSV から読み込む（GitHub 上の CSV を使う想定）
csv_url = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
)
dataset = ray.data.read_csv(csv_url)

# 高額チップフラグを付与（例：tip / total_bill > 0.2 を 1）
THRESHOLD = 0.2


def create_label(batch):
    import pandas as pd

    df = batch if isinstance(batch, pd.DataFrame) else pd.DataFrame(batch)
    df["tip_ratio"] = df["tip"] / df["total_bill"]
    df["is_high_tip"] = (df["tip_ratio"] > THRESHOLD).astype(int)
    return df


dataset = dataset.map_batches(create_label)

# 学習 / 検証に分割
train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)

# 分散処理用にブロック数を調整（Pi クラスタなら 2〜4 くらい）
train_dataset = train_dataset.repartition(num_blocks=2)
valid_dataset = valid_dataset.repartition(num_blocks=2)

# Min-Max スケーラー（使わないなら消してOK）
preprocessor = MinMaxScaler(columns=["total_bill", "tip", "size"])

LABEL_COL = "is_high_tip"
FEATURE_COLS = ["total_bill", "tip", "size"]  # シンプルに数値だけ使う


#########################
# 3. train_loop_per_worker の定義
#########################

def train_loop_per_worker(config):
    """各ワーカーで実行される学習ループ."""
    # 必要な情報は config と session から取る
    label_col = config["label_column"]
    feature_cols = config["feature_columns"]
    params = config["params"]
    num_boost_round = config["num_boost_round"]

    # Ray AIR セッションからデータセット shard を取得
    train_shard = session.get_dataset_shard("train").to_pandas()
    valid_shard = session.get_dataset_shard("valid").to_pandas()

    # pandas → XGBoost の DMatrix に変換
    dtrain = xgb.DMatrix(train_shard[feature_cols], label=train_shard[label_col])
    dvalid = xgb.DMatrix(valid_shard[feature_cols], label=valid_shard[label_col])

    # 学習
    evals = [(dtrain, "train"), (dvalid, "valid")]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
    )

    # 検証精度を計算
    preds = model.predict(dvalid)
    pred_label = (preds >= 0.5).astype(int)
    accuracy = float(
        (pred_label == valid_shard[label_col].to_numpy()).mean()
    )

    # メトリクスを Ray に報告
    session.report(
        {
            "valid_accuracy": accuracy,
            "num_boost_round": num_boost_round,
        }
    )


#########################
# 4. トレーナの定義 & 実行
#########################

config = {
    "label_column": LABEL_COL,
    "feature_columns": FEATURE_COLS,
    "params": {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "tree_method": "approx",  # CPU 用
    },
    "num_boost_round": 200,
}

trainer = XGBoostTrainer(
    train_loop_per_worker=train_loop_per_worker,
    scaling_config=ScalingConfig(
        num_workers=1,    # Pi クラスタのサイズに応じて 1 or 2 に調整
        use_gpu=False,
    ),
    datasets={"train": train_dataset, "valid": valid_dataset},
    # preprocessor は Ray 2.52 の XGBoostTrainer では
    # サポートされてない可能性が高いので、使うなら
    # train_loop 内で自前で前処理するのが安全
    # preprocessor=preprocessor,
)

result = trainer.fit()

print("==== Training Finished ====")
print(f"valid_accuracy = {result.metrics.get('valid_accuracy')}")

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "asset_telemetry.csv"
PLAYBOOK_PATH = BASE / "data" / "fault_playbooks.json"


def load_playbooks():
    with open(PLAYBOOK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Expected power baseline per asset type
    df["expected_power_mw"] = np.where(
        df["asset_id"].str.startswith("SOLAR"),
        0.0019 * df["irradiance_wm2"],
        np.maximum(0.0, 0.44 * df["wind_speed_ms"] - 1.30),
    )
    df["power_residual"] = df["power_mw"] - df["expected_power_mw"]

    # Per-asset rolling z-scores for temperature and vibration
    for col in ["temperature_c", "vibration_mm_s"]:
        grp = df.groupby("asset_id")[col]
        mean = grp.transform(lambda s: s.rolling(6, min_periods=1).mean())
        std = grp.transform(lambda s: s.rolling(6, min_periods=1).std()).replace(0, 1.0).fillna(1.0)
        df[f"{col}_z"] = (df[col] - mean) / std

    return df


def retrieve_playbook(row: pd.Series, playbooks: list) -> dict:
    is_solar = str(row["asset_id"]).startswith("SOLAR")
    if is_solar and row["vibration_mm_s"] > 2.5 and row["temperature_c"] > 42:
        target_id = "SOLAR_NIGHT_THERMAL"
    elif (not is_solar) and row["vibration_mm_s"] > 2.5 and row["power_residual"] < -0.35:
        target_id = "WIND_DRIVETRAIN_DERATE"
    else:
        target_id = "GENERAL_SENSOR_DRIFT"
    return next(p for p in playbooks if p["fault_id"] == target_id)


def format_operator_explanation(row: pd.Series, pb: dict) -> dict:
    summary = (
        f"Asset {row['asset_id']} at timestamp {int(row['timestamp_id'])} (hour {int(row['hour'])}:00) "
        f"recorded temp {row['temperature_c']:.1f}°C, vibration {row['vibration_mm_s']:.2f} mm/s, "
        f"and power {row['power_mw']:.2f} MW (expected {row['expected_power_mw']:.2f} MW)."
    )
    return {
        "summary": summary,
        "fault_code": pb["fault_id"],
        "likely_cause": pb["likely_cause"],
        "recommended_checks": pb["next_checks"],
    }


def run_detection_pipeline():
    raw_df = pd.read_csv(DATA_PATH)
    df = engineer_features(raw_df)
    playbooks = load_playbooks()

    feature_cols = ["temperature_c", "vibration_mm_s", "power_residual", "temperature_c_z", "vibration_mm_s_z"]
    df["iforest_flag"] = 0
    df["anomaly_score"] = 0.0

    # Fit IsolationForest separately per asset type
    for asset_id, idx in df.groupby("asset_id").groups.items():
        model = IsolationForest(contamination=0.03, random_state=42)
        preds = model.fit_predict(df.loc[idx, feature_cols])
        scores = -model.decision_function(df.loc[idx, feature_cols])
        df.loc[idx, "iforest_flag"] = (preds == -1).astype(int)
        df.loc[idx, "anomaly_score"] = scores

    # Hybrid rule + ML flag
    rule_flag = (df["vibration_mm_s"] > 2.5) | (df["power_residual"] < -0.45)
    df["is_anomaly"] = ((df["iforest_flag"] == 1) & rule_flag) | (df["vibration_mm_s"] > 3.0)

    # Benchmark against known planted fault windows (67-69 and 173-174)
    y_true = df["timestamp_id"].isin([67, 68, 69, 173, 174]).astype(int)
    y_pred = df["is_anomaly"].astype(int)
    metrics = {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "total_records": int(len(df)),
        "anomalies_detected": int(y_pred.sum()),
    }

    explanations = {}
    for _, row in df[df["is_anomaly"]].iterrows():
        pb = retrieve_playbook(row, playbooks)
        explanations[int(row["timestamp_id"])] = format_operator_explanation(row, pb)

    return df, metrics, explanations
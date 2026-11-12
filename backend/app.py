from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.pipeline import run_detection_pipeline

app = FastAPI(title="Renewable Asset Fault & Anomaly Detection Copilot")
BASE = Path(__file__).resolve().parents[1]
FRONTEND_HTML = BASE / "frontend" / "index.html"


class Query(BaseModel):
    text: str = ""
    timestamp_id: Optional[int] = None


@app.get("/")
def serve_ui():
    return FileResponse(FRONTEND_HTML)


@app.get("/api/demo")
def demo():
    _, metrics, explanations = run_detection_pipeline()
    return {
        "case_study": "Renewable Asset Fault & Anomaly Detection Copilot",
        "status": "running",
        "metrics": metrics,
        "flagged_timestamps": list(explanations.keys()),
    }


@app.get("/api/telemetry")
def get_telemetry():
    df, metrics, explanations = run_detection_pipeline()
    records = df.round(3).to_dict(orient="records")
    return {
        "metrics": metrics,
        "explanations": explanations,
        "records": records,
    }


@app.post("/api/run")
def run_case(query: Query):
    df, metrics, explanations = run_detection_pipeline()
    text = query.text.lower()

    selected_ts = query.timestamp_id
    if selected_ts is None:
        if "wtg" in text or "wind" in text:
            selected_ts = next((ts for ts in explanations if ts >= 120), None)
        else:
            selected_ts = next(iter(explanations.keys()), None)

    if selected_ts is not None and selected_ts in explanations:
        detail = explanations[selected_ts]
        return {
            "input": query.text,
            "selected_timestamp_id": selected_ts,
            "prediction": "ANOMALY_DETECTED",
            "diagnosis": detail,
            "reasoning_steps": [
                "1. Computed asset-specific expected power curve and rolling 6-hour sensor z-scores.",
                "2. Scored telemetry point with per-asset IsolationForest + physical safety thresholds.",
                f"3. Retrieved SOP playbook '{detail['fault_code']}' matching multi-sensor fault signature.",
            ],
            "metrics": metrics,
            "limitations": "Trained on 240-hour synthetic baseline; requires seasonal retraining and SCADA sensor quality flags in production.",
        }

    return {
        "input": query.text,
        "prediction": "NORMAL",
        "result": "No active fault matched your query.",
        "metrics": metrics,
    }
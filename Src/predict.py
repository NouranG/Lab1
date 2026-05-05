from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
import yaml, os
from typing import Any

load_dotenv()

app = FastAPI()
model = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cfg_path = os.path.join(BASE_DIR, "..", "conf", "cfg.yaml")

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{cfg['model']['repo_name']}.mlflow"
)

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.pyfunc.load_model("models:/credit_model/1")
    print("Model loaded successfully!")

@app.post("/predict")
async def predict(data: dict[str, Any]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    df = pd.DataFrame([data])
    preds = model.predict(df)
    return {"prediction": preds.tolist()}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
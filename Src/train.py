import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pickle
import os
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from dotenv import load_dotenv
import yaml

load_dotenv()

#loading cfg file:

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cfg_path = os.path.join(BASE_DIR, "..", "conf", "cfg.yaml")

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

#setting tracking uri
tracking_uri=  f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{cfg['model']['repo_name']}.mlflow"
mlflow.set_tracking_uri(tracking_uri)


#instantiating mlflow client
client=MlflowClient(tracking_uri=tracking_uri)

  # already exists

#logging models:


mlflow.set_experiment("titanic")



def train_model(X, y, model, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))

    best_auc = -1
    best_model = None

    with mlflow.start_run():
        try:
            client.create_registered_model("credit_model")
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e) and "already exists" not in str(e):
                raise
   

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)

            preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = preds

            fold_auc = roc_auc_score(y_val, preds)
            print(f"Fold {fold+1} AUC:", fold_auc)

            if fold_auc > best_auc:
                best_auc = fold_auc
                best_model = model  

        overall_auc = roc_auc_score(y, oof_preds)

        print("Overall AUC:", overall_auc)

        mlflow.log_metric("overall_auc", float(overall_auc))
        mlflow.log_param("model_type", type(model).__name__)

        #  log + register  model
        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name="credit_model"
        ) # type: ignore[attr-defined]

    return model, oof_preds
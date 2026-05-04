import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pickle
import os
import dagshub
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from dotenv import load_dotenv
import yaml

load_dotenv()

#loading cfg file:
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

#setting tracking uri
tracking_uri=  f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{cfg['model']['repo_name']}.mlflow"
mlflow.set_tracking_uri(tracking_uri)


#instantiating mlflow client
client=mlflow.MlflowClient(tracking_uri=tracking_uri)

#logging models:



def train_model(X, y, model,n_splits=5):

    mlflow.set_experiment("model")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))

    best_auc = -1
    best_model = None

    with mlflow.start_run():

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
                best_model = pickle.dumps(model)
        overall_auc = roc_auc_score(y, oof_preds)


        print("Overall AUC:", overall_auc)

        if best_model is not None:
            os.makedirs("models", exist_ok=True)
            with open("models/model.pkl", "wb") as f:
                f.write(best_model)
            print("Best model saved to models/model.pkl")

        
        mlflow.log_metric("overall_auc", overall_auc)
        mlflow.log_param("model_type", model)

        mlflow.sklearn.log_model(best_model, "model",
        registered_model_name="prod_model")

        return model, oof_preds





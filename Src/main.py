import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from Src.data_loader import load_data
from Src.model import get_model
from Src.preprocessing import DataPreprocessor
from Src.train import train_model

# ------------------
# 1. data loading
# ------------------

train = load_data("/teamspace/studios/this_studio/Titanic/data/train.csv")
test = load_data("/teamspace/studios/this_studio/Titanic/data/test.csv")

test_ids = test["PassengerId"]


# ----------------------
# 2. Splitting X and Y
# ----------------------

X = train.drop(["Survived"], axis=1)
y = train["Survived"]

# --------------------
# 3. preprocessing
# --------------------

preprocess = DataPreprocessor()
X = preprocess.drop_columns(X)
test = preprocess.drop_columns(test)

preprocessor = preprocess.build_pipeline(X)

# ---------------------------
# 4. Train/Validation split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------
# 5. models
# --------------------
models = {}
for model_name in ["rf", "lgb"]:
    model = get_model(model_name)
    model_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

models[model_name] = model_pipeline


# ---------------------
# 6. Model training
# ---------------------

results = {}
best_model_name = None
best_score = 0

for name, pipeline in models.items():

    print(f"\n========== Training {name} ==========")

    trained_model, oof_preds = train_model(X, y, pipeline, n_splits=5)

    score = roc_auc_score(y, oof_preds)

    results[name] = score

    print(f"{name} Final AUC: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model_name = name


print("\nBest Model:", best_model_name)

# ---------------------------
# 7. Retrain best model on full data
# ---------------------------
best_pipeline = models[best_model_name]
best_pipeline.fit(X, y)

# ---------------------------
# 8.Prediction
# ---------------------------
test_preds = best_pipeline.predict(test)

# ---------------------------
# 9. Save predictions
# ---------------------------
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_preds})

submission.to_csv("submission.csv", index=False)

print("\nSubmission saved successfully!")

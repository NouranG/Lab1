from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


def get_model(model_name):
    if model_name == "lgb":

        return LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    elif model_name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    else:
        raise ValueError("Unknown model")

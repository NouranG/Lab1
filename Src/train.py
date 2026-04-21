from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def train_model(X, y, model, n_splits=5):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds
        
        print(f"Fold {fold+1} AUC:", roc_auc_score(y_val, preds))
    
    print("Overall AUC:", roc_auc_score(y, oof_preds))
    
    return model, oof_preds
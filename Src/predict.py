def predict(model, X_test):
    return model.predict_proba(X_test)[:, 1]
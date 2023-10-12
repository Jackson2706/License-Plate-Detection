from joblib import load

loaded_model, loaded_scaler = load("Weights/clf_model_and_scaler_feature.pkl")
print(loaded_model)
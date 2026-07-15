import joblib

model = joblib.load("models/house_prediction.pkl")

print(type(model))
print(model)

if hasattr(model, "feature_names_in_"):
    print("Features:", model.feature_names_in_)
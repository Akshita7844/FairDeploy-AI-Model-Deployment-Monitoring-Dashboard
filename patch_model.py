# patch_model.py
import joblib
import os

model_path = "models/partner_model.pkl"
model = joblib.load(model_path)

# Patch always and force re-save
model.monotonic_cst = None  # Safe to overwrite
joblib.dump(model, model_path)

print("Model forcibly re-saved with dummy monotonic_cst ✅")
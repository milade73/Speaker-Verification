import joblib

# Change this to the actual path of your original model
model_path = "/Users/mmli/Desktop/Final_multi_modal2/Video/weights/Gradient Boosting.pkl"
resaved_path = "/Users/mmli/Desktop/Final_multi_modal2/Video/model_resaved.pkl"

model = joblib.load(model_path)
print("✅ Model loaded successfully.")

joblib.dump(model, resaved_path)
print(f"✅ Model re-saved at: {resaved_path}")
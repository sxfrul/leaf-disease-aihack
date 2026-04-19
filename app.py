import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Leaf Disease Detection API")

# --- CORS Setup ---
# This allows your frontend (UniNet, a dashboard, or mobile app) to talk to the API without browser security blocking it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
MODEL_PATH = 'model.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Image Preprocessing ---
# IMPORTANT: Change this to the size your model expects!
TARGET_SIZE = (224, 224) 

def process_image(image_bytes):
    try:
        # Open the image from the uploaded bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Ensure it's in RGB format (removes alpha channels if PNG)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Resize to the target size expected by your model
        image = image.resize(TARGET_SIZE)
        
        # Convert to numpy array and add the batch dimension
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize if your model requires it (e.g., dividing by 255.0)
        # img_array = img_array / 255.0 
        
        return img_array
    except Exception as e:
        raise ValueError(f"Invalid image format: {e}")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Leaf Disease Detection API is running. Send a POST request to /predict with an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    
    # Read the file contents into memory
    contents = await file.read()
    
    try:
        # Preprocess the image
        processed_image = process_image(contents)
        
        # Run inference
        predictions = model.predict(processed_image)
        
        # Format the output based on your model's design
        # If it's multi-class classification, you usually want the argmax
        predicted_class_index = int(np.argmax(predictions, axis=-1)[0])
        confidence = float(np.max(predictions[0]))
        
        return {
            "filename": file.filename,
            "predicted_class_index": predicted_class_index,
            "confidence": confidence,
            "raw_predictions": predictions.tolist()
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
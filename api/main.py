from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import keras


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL = tf.keras.models.load_model('models/potato-model_15-epochs.keras')

# Define the class names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# @app.get("/ping")
# async def ping():
#     return {"ping": "pong!"}

# Read file and convert pillow image to numpy array
def read_imagefile(data) -> np.ndarray:
  image = np.array(Image.open(io.BytesIO(data)))
  return image

# Predict the class of the image
@app.post("/predict")
async def predict(
  file: UploadFile = File(...),
):
  image = read_imagefile(await file.read())
  img_batch = np.expand_dims(image, 0)
  predictions = MODEL.predict(img_batch)
  predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
  confidence = round((np.max(predictions[0])), 2)
  return {"class": predicted_class, "confidence": confidence}
    

if __name__ == "__main__":
  uvicorn.run(app, host='localhost', port=8000)
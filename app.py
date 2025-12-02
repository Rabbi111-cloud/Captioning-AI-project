from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return {"caption": caption}

# Run locally: uvicorn app:app --reload

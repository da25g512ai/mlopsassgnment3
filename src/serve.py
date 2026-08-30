import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from model import get_model
from dataset import get_transforms

app = FastAPI()
device = torch.device("cpu")
model = None
transforms = get_transforms(train=False)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

@app.on_event("startup")
def load_model():
    global model
    try:
        m = get_model(architecture="resnet18", num_classes=10)
        checkpoint = torch.load("/app/checkpoints/classifier_v1.pt", map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        m.load_state_dict(clean_state_dict)
        m.eval().to(device)
        model = m
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/health")
def health():
    if model is not None:
        return JSONResponse(status_code=200, content={"status": "healthy"})
    raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0).tolist()
    
    predicted_idx = int(torch.argmax(outputs, dim=1).item())
    return {"prediction": CLASSES[predicted_idx], "probabilities": {CLASSES[i]: round(probs[i], 4) for i in range(10)}}
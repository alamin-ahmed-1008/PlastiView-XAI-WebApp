import os
import time
import base64
import io
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import cv2

app = FastAPI(title="PlastiView XAI API")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine device (MPS, CUDA, or CPU)
device = torch.device("cpu")
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")

print(f"Using device for inference: {device}")

# Global model dictionary
loaded_models = {}

def load_model(path, name):
    print(f"Loading {name} model from {path}...")
    try:
        # Standard torchvision Faster R-CNN ResNet-50 FPN with 2 classes (background + plastic)
        model = fasterrcnn_resnet50_fpn(num_classes=2, weights=None)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        loaded_models[name] = model
        print(f"Successfully loaded {name} model.")
    except Exception as e:
        print(f"Error loading {name} model: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    print("Application started. Models will be loaded on demand when inference is requested.")

# Hook to capture features for Grad-CAM
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.hook = self.model.backbone.body.layer4.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # Output shape is [1, 2048, H_feat, W_feat]
        self.features = output

    def remove(self):
        self.hook.remove()

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    model_id: str = Form("finetuned"),
    threshold: float = Form(0.5)
):
    start_time = time.time()
    
    # Lazy load the selected model if not already loaded in memory
    if model_id not in loaded_models:
        models_dir = "models"
        if model_id == "base":
            model_path = os.path.join(models_dir, "fasterrcnn.pth")
        elif model_id == "finetuned":
            model_path = os.path.join(models_dir, "finetuned.pth")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model ID: '{model_id}'")
            
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found at: {model_path}")
            
        load_model(model_path, model_id)
        
        if model_id not in loaded_models:
            raise HTTPException(status_code=500, detail=f"Failed to load model: '{model_id}'")
            
    model = loaded_models[model_id]
    
    try:
        # Read uploaded image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        orig_width, orig_height = image.size
        
        # Preprocess for model input
        # PIL Image to tensor with values in [0, 1]
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(device)
        
        # Setup hook to capture ResNet layer 4 features
        extractor = FeatureExtractor(model)
        
        # Run inference
        with torch.no_grad():
            predictions = model([image_tensor])
            
        # Extract features and remove hook immediately
        captured_features = extractor.features
        extractor.remove()
        
        inference_time = time.time() - start_time
        
        # Parse predictions
        prediction = predictions[0]
        pred_boxes = prediction["boxes"].cpu().numpy()
        pred_labels = prediction["labels"].cpu().numpy()
        pred_scores = prediction["scores"].cpu().numpy()
        
        # Filter by threshold and label (class 1 is target microplastic)
        keep_indices = np.where((pred_scores >= threshold) & (pred_labels == 1))[0]
        
        filtered_boxes = []
        for idx in keep_indices:
            box = pred_boxes[idx]
            score = pred_scores[idx]
            
            # Convert absolute coordinates (x1, y1, x2, y2) to percentage coordinates (x, y, w, h)
            # x, y = top-left corner
            x_min, y_min, x_max, y_max = box
            
            x_pct = (x_min / orig_width) * 100
            y_pct = (y_min / orig_height) * 100
            w_pct = ((x_max - x_min) / orig_width) * 100
            h_pct = ((y_max - y_min) / orig_height) * 100
            
            filtered_boxes.append({
                "id": int(idx),
                "x": float(np.clip(x_pct, 0, 100)),
                "y": float(np.clip(y_pct, 0, 100)),
                "w": float(np.clip(w_pct, 0, 100)),
                "h": float(np.clip(h_pct, 0, 100)),
                "confidence": f"{float(score):.3f}"
            })
            
        # Sort boxes by confidence descending
        filtered_boxes = sorted(filtered_boxes, key=lambda b: float(b["confidence"]), reverse=True)
        
        # Generate Grad-CAM heatmaps
        # Convert captured_features to a heatmap
        if captured_features is not None:
            # Take mean across channel dimension [1, 2048, H, W] -> [H, W]
            feat_map = torch.mean(captured_features, dim=1).squeeze().cpu().numpy()
            
            # Apply ReLU (only positive activations)
            feat_map = np.maximum(feat_map, 0)
            
            # Normalize to [0, 1]
            map_max = feat_map.max()
            if map_max > 0:
                feat_map = feat_map / map_max
                
            # Convert to uint8 format
            heatmap_uint8 = np.uint8(255 * feat_map)
            
            # Resize heatmap to match original image size
            heatmap_resized = cv2.resize(heatmap_uint8, (orig_width, orig_height))
            
            # Apply ColorMap JET
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Convert original image to grayscale BGR for the mockup style blending
            # image_np is in RGB format, so convert to grayscale
            orig_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            orig_gray_bgr = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
            
            # Blend grayscale original and the color heatmap
            # Jet colormap returns BGR. So let's blend in BGR space
            blended = cv2.addWeighted(orig_gray_bgr, 0.4, heatmap_color, 0.6, 0)
            
            # Convert blended from BGR to RGB before encoding
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            
            # Encode blended image to PNG bytes
            blended_pil = Image.fromarray(blended_rgb)
            img_byte_arr = io.BytesIO()
            blended_pil.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Base64 encode
            gradcam_base64 = "data:image/png;base64," + base64.b64encode(img_byte_arr).decode('utf-8')
        else:
            # Fallback if no features captured (should not happen)
            gradcam_base64 = ""
            
        return {
            "particleCount": len(filtered_boxes),
            "inferenceTime": f"{inference_time:.2f}",
            "boxes": filtered_boxes,
            "gradcam": gradcam_base64
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/api/status")
async def get_status():
    return {
        "status": "connected",
        "device": str(device),
        "models": list(loaded_models.keys())
    }

# Serve frontend HTML directly or static folder if created
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join("templates", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Frontend index.html template not found. Please create it under templates/index.html.</h3>"

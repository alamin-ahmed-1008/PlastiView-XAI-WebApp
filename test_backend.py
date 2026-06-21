import asyncio
import io
import os
import numpy as np
from PIL import Image, ImageDraw
import torch

# Create a dummy image representing a microscopy capture
# with some simulated particles (dark circles on light background)
def create_dummy_microscopy_image():
    # 512x512 RGB image
    img = Image.new('RGB', (512, 512), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    
    # Draw some random circles resembling microplastics
    # Particle 1
    draw.ellipse([100, 150, 140, 190], fill=(50, 60, 80), outline=(20, 30, 40))
    # Particle 2
    draw.ellipse([300, 320, 350, 370], fill=(70, 50, 60), outline=(40, 20, 30))
    # Particle 3
    draw.ellipse([220, 180, 280, 240], fill=(40, 70, 50), outline=(10, 40, 20))
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

async def test_inference():
    print("Testing backend startup and model loading...")
    # Import the FastAPI app
    from app import app, startup_event, analyze_image
    from fastapi import UploadFile
    
    # Manually trigger startup event to load models
    await startup_event()
    
    print("\nSimulating image upload...")
    dummy_img_bytes = create_dummy_microscopy_image()
    
    # Create fastapi UploadFile object
    upload_file = UploadFile(
        filename="test_microscopy.png", 
        file=dummy_img_bytes
    )
    
    print("\nRunning inference using 'finetuned' model...")
    # Call the api endpoint function directly
    result = await analyze_image(
        file=upload_file,
        model_id="finetuned",
        threshold=0.30
    )
    
    print("\n--- INFERENCE RESULTS ---")
    print(f"Status: SUCCESS")
    print(f"Inference Time: {result['inferenceTime']} seconds")
    print(f"Microplastics Detected: {result['particleCount']}")
    print(f"Detections List:")
    for box in result["boxes"]:
        print(f"  - Box: x={box['x']:.2f}%, y={box['y']:.2f}%, w={box['w']:.2f}%, h={box['h']:.2f}%, conf={box['confidence']}")
        
    print(f"\nGrad-CAM heatmaps generated successfully.")
    print(f"Grad-CAM base64 string length: {len(result['gradcam'])} characters")
    assert len(result['gradcam']) > 0, "Grad-CAM output should not be empty"
    print("\nAll backend checks passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_inference())

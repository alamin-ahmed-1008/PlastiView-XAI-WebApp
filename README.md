# PlastiView XAI - Setup & Running Guide

This guide provides instructions on how to create a Python virtual environment, install dependencies, and run the PlastiView XAI Web Application.

---

## Prerequisites

- **Python 3.10+** installed on your system.
- The two Faster R-CNN model files placed in the `models/` directory:
    - `fasterrcnn.pth` (Base Model)
    - `finetuned.pth` (Fine-Tuned Model)

---

## 1. Create a Python Virtual Environment

A virtual environment helps isolate project dependencies from your system-wide Python installation.

Open your terminal or command prompt, navigate to the project directory, and run the following command:

```bash
python3 -m venv venv
```

_Note: On some systems, you might need to use `python` instead of `python3`._

---

## 2. Activate the Virtual Environment

Before installing dependencies or running the app, you must activate the virtual environment:

### On macOS / Linux

```bash
source venv/bin/activate
```

### On Windows (Command Prompt)

```cmd
venv\Scripts\activate
```

### On Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

Once activated, you will see `(venv)` prepended to your command line prompt.

---

## 3. Install Project Dependencies

With the virtual environment active, run the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

This installs:

- **FastAPI**: The web framework for our backend API.
- **Uvicorn**: The ASGI server to run FastAPI.
- **PyTorch & Torchvision**: The deep learning frameworks to load and execute the Faster R-CNN model.
- **OpenCV & Pillow**: Libraries for image processing, color mapping, and Grad-CAM generation.
- **Jinja2 & Multipart**: Packages for serving HTML templates and handling upload requests.

---

## 4. Run the Web Application

To start the FastAPI web server, run uvicorn from the root project directory:

```bash
uvicorn app:app --reload
```

- `app:app` refers to the `app.py` file and the `app` FastAPI instance inside it.
- `--reload` enables auto-reloading, which restarts the server automatically when code changes.

On startup, uvicorn will load both PyTorch models. This might take 3-5 seconds. You will see output similar to this:

```text
Using device for inference: mps
Loading base model from models/fasterrcnn.pth...
Successfully loaded base model.
Loading finetuned model from models/finetuned.pth...
Successfully loaded finetuned model.
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn server running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 5. Access the Web Dashboard

Open your web browser and navigate to:

👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

You can now:

1. Select the model (Base or Client-Adapted).
2. Set the confidence threshold using the slider.
3. Upload/drag-and-drop a microscopy image.
4. Run inference and view the detected microplastic bounding boxes, confidence scores, and Grad-CAM Layer 4 heatmaps.

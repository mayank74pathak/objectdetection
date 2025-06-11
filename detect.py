# inference/detect.py
import cv2
import numpy as np
from pathlib import Path
import torch # <--- CRITICAL MISSING IMPORT!
import sys   # <--- Also add sys for path manipulation if needed for robustness (as discussed previously)

# Add the project root to the Python path to allow importing from other directories
# This is generally a good practice for internal project imports, though app.py's
# sys.path modification is what directly enables the import into app.py.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))


class Detector:
    def __init__(self, model_path):
        if not model_path.exists(): # Added a check for model existence
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        print(f"Loading model from: {model_path}") # Added print for debugging
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), force_reload=True)
        self.model.eval() # Set model to evaluation mode (good practice for inference)
        print("Model loaded successfully.")

        # Define class names - ensure these match your dataset.yaml
        self.classes = ['pen', 'bottle'] # IMPORTANT: Make sure this matches dataset.yaml

    def detect(self, img):
        """Run detection on single image (numpy array BGR format)"""
        # YOLOv5 expects RGB images, so convert if input is BGR from OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img # Assume it's already RGB or grayscale

        # Perform inference
        results = self.model(img_rgb) # Pass the RGB image
        return results

    def draw_boxes(self, img, results):
        """Draw bounding boxes on image (expects BGR image)"""
        # Ensure image is writeable for drawing
        if not img.flags['WRITEABLE']:
            img = img.copy()

        # results.xyxy[0] contains detections for the first image in the batch
        # Each detection is [x1, y1, x2, y2, confidence, class_id]
        if results.xyxy[0].shape[0] > 0:
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                
                # Get class label safely
                class_id = int(cls)
                if class_id < len(self.classes):
                    label = f"{self.classes[class_id]} {conf:.2f}"
                else:
                    label = f"Unknown {conf:.2f}" # Fallback for unexpected class ID
                
                color = (0, 255, 0) # Green color for bounding box

                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)

                # Draw label text
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return img
# app.py
import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile
import cv2
from pathlib import Path
import sys
import os

# --- STREAMLIT CLOUD COMPATIBILITY START ---
# Get the current script directory (where app.py is located)
script_dir = Path(__file__).resolve().parent

# In Streamlit Cloud, the project root is the same as script directory
# since app.py is in the root of the repository
project_root = script_dir

# Add the project root to the Python path for imports
sys.path.append(str(project_root))
# --- STREAMLIT CLOUD COMPATIBILITY END ---

from inference.detect import Detector

# Page config
st.set_page_config(
    page_title="Pen & Bottle Detector",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.upload-container {
    border: 2px dashed #ccc;
    border-radius: 5px;
    padding: 30px;
    text-align: center;
    margin: 20px 0;
    cursor: pointer;
}
.upload-container:hover {
    border-color: #1a73e8;
}
.detection-results {
    margin-top: 30px;
}
.st-emotion-cache-1r6dmym {
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# Load model with multiple path options for compatibility
@st.cache_resource
def load_model():
    # Try multiple possible paths for the model file
    possible_model_paths = [
        project_root / "best.pt",                    # Same directory as app.py
        project_root / "models" / "best.pt",         # In a models subdirectory
        project_root / "weights" / "best.pt",        # In a weights subdirectory
        Path("best.pt"),                             # Current working directory
        Path("./best.pt"),                           # Explicit current directory
    ]
    
    model_path = None
    for path in possible_model_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        # Show all attempted paths for debugging
        attempted_paths = "\n".join([f"- {path}" for path in possible_model_paths])
        st.error(f"""
        Model file 'best.pt' not found. Attempted paths:
        {attempted_paths}
        
        Please ensure 'best.pt' is uploaded to your repository root directory.
        """)
        
        # Additional debugging info for Streamlit Cloud
        st.info(f"""
        **Debug Info:**
        - Current working directory: {os.getcwd()}
        - Script directory: {script_dir}
        - Project root: {project_root}
        - Files in project root: {list(project_root.glob('*')) if project_root.exists() else 'Project root not found'}
        """)
        return None

    try:
        st.info(f"Loading model from: {model_path}")
        detector_instance = Detector(model_path)
        st.success("Model loaded successfully!")
        return detector_instance
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("""
        Possible solutions:
        1. Ensure 'best.pt' is in the repository root
        2. Check that the model file is not corrupted
        3. Verify all dependencies are properly installed
        """)
        return None

# Initialize detector
detector = load_model()

# Main UI
st.title("🔍 Pen & Bottle Detection System")

if detector is None:
    st.warning("⚠️ Model not loaded. Please check the error messages above.")
    st.stop()

# Input selection
input_type = st.radio("Select input type:", ("Image", "Video", "Webcam"))

if input_type == "Image":
    st.markdown("""
    <div class="upload-container">
        <h3>📸 Upload Image</h3>
        <p>Drag and drop or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            with st.spinner("🔍 Detecting objects..."):
                results = detector.detect(img_bgr)
                annotated_img_bgr = detector.draw_boxes(img_bgr.copy(), results)
                annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(annotated_img_rgb, caption="Detection Results", use_column_width=True)
            
            # Display results table
            if results.xyxy[0].shape[0] > 0:
                st.markdown("### 📊 Detection Results")
                detections = []
                for *box, conf, cls in results.xyxy[0].cpu().numpy():
                    detections.append({
                        "Object": detector.classes[int(cls)],
                        "Confidence": f"{conf:.2%}",
                        "Location (x1, y1, x2, y2)": f"({int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])})"
                    })
                
                st.dataframe(detections, use_container_width=True)
            else:
                st.info("ℹ️ No objects detected.")
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

elif input_type == "Video":
    st.markdown("""
    <div class="upload-container">
        <h3>🎥 Upload Video</h3>
        <p>Drag and drop or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "webm"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            st.video(uploaded_file)
            
            # Process video button
            if st.button("🎬 Process Video"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                
                st.markdown("### 🎥 Video Detection Results")
                stframe = st.empty()
                
                cap = cv2.VideoCapture(tfile.name)
                
                if not cap.isOpened():
                    st.error("❌ Error: Could not open video file.")
                    if Path(tfile.name).exists():
                        Path(tfile.name).unlink()
                    st.stop()

                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress_bar = st.progress(0)
                
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        results = detector.detect(frame)
                        annotated_frame = detector.draw_boxes(frame.copy(), results)
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        stframe.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                        
                        frame_count += 1
                        progress_bar.progress(min(1.0, frame_count / total_frames))

                    st.success("✅ Video processing complete!")
                except Exception as e:
                    st.error(f"❌ Error during video processing: {e}")
                finally:
                    cap.release()
                    if Path(tfile.name).exists():
                        Path(tfile.name).unlink()
                    progress_bar.empty()
                    
        except Exception as e:
            st.error(f"Error processing video: {e}")

elif input_type == "Webcam":
    st.warning("⚠️ **Webcam Limitations in Streamlit Cloud:**")
    st.info("""
    - Webcam access is **not supported** in Streamlit Cloud due to browser security restrictions
    - This feature only works when running locally
    - For cloud deployment, please use the Image or Video upload options instead
    """)
    
    st.markdown("### 🖥️ Local Development Instructions")
    st.code("""
    # To run locally with webcam support:
    streamlit run app.py
    """, language="bash")
    
    # Show webcam interface anyway for demo purposes (won't work in cloud)
    if st.checkbox("Show webcam interface (Local only)"):
        if "webcam_active" not in st.session_state:
            st.session_state.webcam_active = False
        
        available_camera_indices = [0, 1, 2]  # Common indices
        selected_camera_index = st.selectbox(
            "Select Camera Device (Index):",
            options=available_camera_indices,
            help="This only works when running locally"
        )

        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("🎥 Start Webcam")
        with col2:
            stop_button = st.button("⏹️ Stop Webcam")

        if start_button:
            st.error("❌ Webcam not available in Streamlit Cloud environment")
        if stop_button:
            st.session_state.webcam_active = False

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🔍 Pen & Bottle Detection System</strong></p>
    <p><em>Powered by YOLOv5 & Streamlit</em></p>
</div>
""", unsafe_allow_html=True)

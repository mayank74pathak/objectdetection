# app/app.py
import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile
import cv2
from pathlib import Path
import sys # <-- Add this import

# --- IMPORTANT CHANGE START ---
# Add the project root to the Python path
# This allows app.py to correctly import modules from sibling directories like 'inference'
script_dir = Path(__file__).resolve().parent # This is '.../objectdetection/app'
project_root = script_dir.parent             # This is '.../objectdetection'
sys.path.append(str(project_root))
# --- IMPORTANT CHANGE END ---

from inference.detect import Detector # This import should now work

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
.st-emotion-cache-1r6dmym { /* Target for specific Streamlit element for radio buttons */
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    # Make sure this path is correct based on your training output
    # You mentioned 'pen_bottle_detection2' was the actual output folder
    # So, let's use that here for clarity and correctness.
    model_path = project_root / best.pt
    
    if not model_path.exists():
        st.error(f"Model file not found at: {model_path}. Please train the model first and verify the path.")
        return None

    try:
        detector_instance = Detector(model_path)
        return detector_instance
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Ensure all dependencies are installed and the model has been trained successfully.")
        return None

detector = load_model()

# Main UI
st.title("Pen & Bottle Detection System")

if detector is None:
    st.stop() # Stop app execution if model loading failed

# Input selection
input_type = st.radio("Select input type:", ("Image", "Video", "Webcam"))

if input_type == "Image":
    st.markdown("""
    <div class="upload-container">
        <h3>Upload Image</h3>
        <p>Drag and drop or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Convert PIL Image to numpy array (RGB)
        img_array = np.array(image)
        # Convert RGB to BGR for OpenCV functions in Detector (if it expects BGR)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Detecting objects..."):
            results = detector.detect(img_bgr) # Pass BGR image for consistent OpenCV usage in draw_boxes
            annotated_img_bgr = detector.draw_boxes(img_bgr.copy(), results)
            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for Streamlit

            st.markdown("### Detection Results")
            st.image(annotated_img_rgb, caption="Detection Results", use_column_width=True)
            
            # Display results table
            if results.xyxy[0].shape[0] > 0:
                detections = []
                for *box, conf, cls in results.xyxy[0].cpu().numpy():
                    detections.append({
                        "Object": detector.classes[int(cls)],
                        "Confidence": f"{conf:.2f}",
                        "Location (x1, y1, x2, y2)": f"({int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])})"
                    })
                
                st.table(detections)
            else:
                st.info("No objects detected.")

elif input_type == "Video":
    st.markdown("""
    <div class="upload-container">
        <h3>Upload Video</h3>
        <p>Drag and drop or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "webm"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.video(uploaded_file, format="video/mp4", start_time=0) # Show original video
        
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close() # Close file handle after writing
        
        st.markdown("### Video Detection Results")
        stframe = st.empty() # Placeholder for video frames
        
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            if Path(tfile.name).exists():
                Path(tfile.name).unlink() # Clean up temp file
            st.stop()

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = cap.get(cv2.CAP_PROP_FPS) # Not used, can remove

        progress_bar = st.progress(0)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = detector.detect(frame)
                annotated_frame = detector.draw_boxes(frame.copy(), results) # Pass a copy to draw on
                
                # Streamlit expects RGB, OpenCV reads BGR
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                
                frame_count += 1
                progress_bar.progress(min(100, int((frame_count / total_frames) * 100)))

            st.success("Video processing complete!")
        except Exception as e:
            st.error(f"An error occurred during video processing: {e}")
        finally:
            cap.release()
            if Path(tfile.name).exists():
                Path(tfile.name).unlink() # Clean up temp file
            progress_bar.empty() # Clear progress bar

elif input_type == "Webcam":
    st.warning("Webcam feature requires running locally with camera permissions. This will block the Streamlit app until stopped.")
    
    # Use session state to manage webcam activity
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False
    
    # --- MODIFICATION START ---
    # Attempt to find available camera indices
    # This loop tries to open and close cameras quickly to find valid indices
    available_camera_indices = []
    # Test up to 10 indices, typically cameras are at 0, 1, 2, etc.
    for i in range(10): 
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            available_camera_indices.append(i)
            temp_cap.release() # Release immediately
        else:
            # If we fail to open a camera, it's possible subsequent indices also fail,
            # especially if there's a gap in device IDs or if it's the end of devices.
            # You can uncomment this break if you want to stop testing after the first failure.
            # break 
            pass # Keep trying other indices as there might be gaps (e.g., 0 fails, 1 works)

    if not available_camera_indices:
        st.error("No webcam devices found. Please ensure a camera is connected and not in use.")
        # If no cameras found, hide controls and exit
        st.session_state.webcam_active = False # Ensure it's off
        st.stop() # Stop the Streamlit script here if no cameras.

    # Streamlit widget to select the camera index
    # Set default index to 0 if available, otherwise the first found index
    default_selected_index = available_camera_indices[0] if available_camera_indices else 0
    selected_camera_index = st.selectbox(
        "Select Camera Device (Index):",
        options=available_camera_indices,
        index=available_camera_indices.index(default_selected_index) if default_selected_index in available_camera_indices else 0,
        help="Select the numerical index corresponding to your desired webcam. Common values are 0, 1, 2."
    )
    # --- MODIFICATION END ---

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Webcam")
    with col2:
        stop_button = st.button("Stop Webcam")

    # If start button is pressed and webcam is not already active
    if start_button and not st.session_state.webcam_active:
        st.session_state.webcam_active = True
        st.rerun() # Rerun the app to enter the webcam loop
    
    # If stop button is pressed
    if stop_button:
        st.session_state.webcam_active = False
        st.rerun() # Rerun the app to exit the webcam loop

    if st.session_state.webcam_active:
        st.markdown("### Live Webcam Detection")
        stframe = st.empty() # Placeholder for live feed
        
        # Use the selected_camera_index here
        cap = cv2.VideoCapture(selected_camera_index) 

        if not cap.isOpened():
            st.error(f"Failed to access webcam at index {selected_camera_index}. It might be in use or not available.")
            st.session_state.webcam_active = False # Disable webcam if it fails
            st.stop() # Stop further execution if webcam fails

        try:
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from webcam. Stream ended or camera disconnected.")
                    st.session_state.webcam_active = False
                    break # Exit loop
                
                results = detector.detect(frame)
                annotated_frame = detector.draw_boxes(frame.copy(), results)
                
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                
                # Small delay to reduce CPU usage, optional but can be useful on some systems
                # time.sleep(0.01) # Requires 'import time' at top
                # cv2.waitKey(1) # Not needed directly with Streamlit image display

        except Exception as e:
            st.error(f"An error occurred during webcam processing: {e}")
        finally:
            cap.release() # Release webcam resource
            st.session_state.webcam_active = False # Ensure state is reset

# Footer
st.markdown("---")
st.markdown("**Pen & Bottle Detection System** | Created for Object Detection Assignment")

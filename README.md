# objectdetection
This repository hosts a Streamlit web application for real-time detection of pens and bottles in images and videos. The system leverages a custom-trained YOLOv5 model to accurately identify these objects.

📊 Dataset Information
The dataset used for training and testing this object detection model was compiled from diverse sources, primarily Google Images and Unsplash.com.

Training Set: A total of 400 images, split equally between:

200 images of pens

200 images of bottles

Testing Set: A total of 100 images, comprising:

50 images of pens

50 images of bottles

🧠 Model Choice: YOLOv5s
For this project, YOLOv5s (YOLOv5 small) was selected as the object detection model.

Why YOLOv5s?

Efficiency: YOLOv5s is the smallest and fastest variant within the YOLOv5 family. This makes it an excellent choice for applications requiring real-time inference and deployment on devices with limited computational resources, such as typical CPU-based Streamlit Cloud instances or edge devices.

Balanced Performance: Despite its small size, YOLOv5s offers a good balance between speed and accuracy for many common detection tasks. It serves as a solid baseline for custom object detection projects, allowing for quick iterations and initial validation.

Ease of Use: The ultralytics framework, which YOLOv5 is part of, provides a user-friendly interface for training and inference, simplifying the development workflow.

🛠️ Challenges Faced and Solutions
Developing and deploying this object detection system involved overcoming a few key challenges:

1. Training Speed and Hardware Requirements
Challenge: Training the YOLOv5s model, especially at an image size of 640 pixels, was significantly slow on the available hardware. Object detection models are computationally intensive, and effective training often requires powerful GPUs.

Solution: For the initial training, a balance was struck with the available resources. For future iterations, leveraging cloud-based GPU instances (e.g., Google Colab Pro, AWS, GCP) or higher-end local GPUs would dramatically accelerate the training process and allow for more extensive experimentation.

2. Optimizing Model Performance (Especially for 'Bottle' Class)
Challenge: The model's performance, particularly for the 'bottle' class, was suboptimal after 30 epochs of training with YOLOv5s. The metrics show a lower recall and mAP50 for bottles compared to pens, indicating underfitting or insufficient learning for this class.

Initial Metrics (30 epochs, YOLOv5s):

all: P=0.706, R=0.737, mAP50=0.767, mAP50-95=0.506

pen: P=0.690, R=0.955, mAP50=0.841, mAP50-95=0.635

bottle: P=0.722, R=0.520, mAP50=0.693, mAP50-95=0.378

Solution & Next Steps:

Increased Epochs: The primary solution is to train for a significantly higher number of epochs (e.g., 100-300 or more) to allow the model to fully converge and learn the features of both classes, especially 'bottle'. 30 epochs is often too few for complex object detection tasks.

Data Augmentation: Implementing more aggressive or varied data augmentation techniques during training can help the model generalize better to unseen variations of bottles (different shapes, angles, lighting).

Hyperparameter Tuning: Fine-tuning hyperparameters such as learning rate, batch size, and optimizer settings can lead to improved convergence and better performance.

Larger Model Variants (YOLOv5m/l): If, after extended training and hyperparameter tuning, YOLOv5s still does not meet the desired performance for bottles, migrating to a larger YOLOv5 variant like YOLOv5m (medium) or YOLOv5l (large) would be the next logical step. These models have more parameters and layers, enabling them to capture more complex features, albeit at the cost of increased computational requirements and slower inference.

🚀 Performance Metrics
The following metrics were obtained after an initial training run of 30 epochs with YOLOv5s, with results saved to /Users/mayankpathak/Desktop/objectdetectioncopy/training/runs/pen_bottle_detection:

Overall Performance (all classes):

Precision (P): 0.706

Recall (R): 0.737

mAP50: 0.767

mAP50-95: 0.506

Class-Specific Performance:

pen:

Precision (P): 0.690

Recall (R): 0.955

mAP50: 0.841

mAP50-95: 0.635

bottle:

Precision (P): 0.722

Recall (R): 0.520

mAP50: 0.693

mAP50-95: 0.378

As noted, the 'bottle' class exhibited lower recall and mAP compared to 'pen', indicating an area for significant improvement through further training and optimization.

📈 Future Improvements & Recommendations
To enhance the "Pen & Bottle Detection System" further, consider these improvements:

Extended Training: Train the current YOLOv5s model for more epochs (e.g., 100-300) to ensure full convergence and improve detection for both classes, especially 'bottle'.

Dataset Expansion: Gather and annotate a larger and more diverse dataset for both pens and bottles, including varied backgrounds, lighting conditions, occlusions, and object states (e.g., empty/full bottles, pens in different orientations).

Model Scaling: Experiment with larger YOLOv5 models such as YOLOv5m or YOLOv5l if YOLOv5s's capacity proves insufficient after comprehensive training. This will likely yield better accuracy but require more computational power.

Advanced Data Augmentation: Implement more sophisticated data augmentation techniques (e.g., Mosaic, MixUp, Albumentations transforms) to make the model more robust.

Hyperparameter Optimization: Conduct a thorough hyperparameter search for learning rate, batch size, optimizer, and other training parameters using tools like Weights & Biases or Optuna.
 

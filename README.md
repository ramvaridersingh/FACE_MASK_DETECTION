
Face Mask Detection using a Convolutional Neural Network (CNN)

 Overview

This project develops a deep learning model to detect whether a person in an image or a real-time video stream is wearing a face mask. The project consists of two main parts:
1.  Model Training: A Convolutional Neural Network (CNN) is built from scratch and trained on a public dataset of images to learn the features of faces with and without masks.
2.  Real-Time Detection: The trained model is deployed in a Python script that uses OpenCV and a webcam to detect faces and classify them in real-time.


 Features

-   Custom Model Training**: Trains a CNN using TensorFlow and Keras.
-   Data Augmentation**: Uses `ImageDataGenerator` to prepare and augment the     image data for robust training.
-   High Accuracy**: Achieves a validation accuracy of over 93% after training.
-   Real-Time Detection**: Uses OpenCV to capture video from a webcam.
-   Live Alerts: Detects faces, draws bounding boxes, and labels them as "Mask" or "No Mask" with a visual alert for non-compliance.


File Structure

├── Train\_Mask\_Model.ipynb          \ Notebook for training the CNN model from scratch.
├── run\_detector.py                 \ Script for the real-time webcam detection.
├── my\_custom\_mask\_detector.h5      \The model trained by the notebook.
├── face\_detector/                  \ Pre-trained OpenCV model for face detection.
│   ├── deploy.prototxt
│   └── res10\_300x300\_ssd\_iter\_140000.caffemodel
├── data/                           \ Folder for training images (must be downloaded separately).
│   ├── with\_mask/
│   └── without\_mask/
└── README.md                       \ This file.

Note: The `data` folder is not included in this repository. You must download it separately (see instructions below).


 How to Run

Part 1: Training Your Own Model (Optional)

1.  Clone the Repository
   bash
    git clone <your-repo-link>
    cd <repo-folder>
   

2.  Set Up Environment
    Create a virtual environment and install the required packages. You should create a `requirements.txt` file with libraries like `tensorflow`, `opencv-python`, `numpy`, and `matplotlib`.
    bash
    pip install -r requirements.txt
    

3.  Download the Dataset
    Download the image dataset from [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) and place the contents into a `data` folder, as shown in the file structure above.

4.  Run the Jupyter Notebook
    Open and run the `Train_Mask_Model.ipynb` notebook. This will train the model and save it as `my_custom_mask_detector.h5`.

    Part 2: Running the Real-Time Detector

1.  Ensure Models are Present
    Make sure you have the `face_detector` folder and a trained model file (e.g., `my_custom_mask_detector.h5`) in your main project directory.

2.  Run the Script
    Execute the following command in your terminal:
    bash
    python run_detector.py
    
    A window will open showing your webcam feed. Press the **'q'** key to close it.


 Results

The CNN model was trained for 10 epochs and successfully achieved an accuracy of **over 93%** on the validation set, demonstrating its effectiveness in distinguishing between masked and unmasked faces.
apps-fileview.texmex_20250828.00_p3
README.md.txt
Displaying README.md.txt.

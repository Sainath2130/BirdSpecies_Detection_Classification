# BirdSpecies_Detection_Classification
Users can upload bird images, and the application displays the top similar images from the training data along with predicted species and similarity scores.
This GitHub repository contains the code for a desktop application built with Tkinter that performs bird species identification using deep learning techniques. The application leverages pre-extracted image features from a deep convolutional neural network (specifically Inception V3 trained on the iNaturalist dataset) and a Logistic Regression classifier for species prediction.

**Key Features:**

* **Image Upload:** Allows users to upload a bird image from their local file system.
* **Feature Extraction (Pre-computed):** Utilizes pre-computed image features to speed up the classification process. The features were extracted using the Inception V3 model trained on the iNaturalist dataset.
* **K-Nearest Neighbor Search:** Employs a KD-Tree to find the K most similar bird images in the training dataset based on the extracted features.
* **Logistic Regression Classification:** Uses a pre-trained Logistic Regression model to predict the bird species based on the features of the uploaded image.
* **Display Similar Images:** Shows the K most similar bird images from the training set along with their predicted species names and similarity scores (based on Euclidean distance in the feature space).
* **Score Visualization:** Presents a bar graph visualizing the similarity scores for the top K predicted species.

**Technical Details:**

* **Deep Learning Model:** Inception V3 (pre-trained on iNaturalist).
* **Feature Extraction:** Features extracted from the chosen pre-trained model.
* **Classifier:** Logistic Regression.
* **Nearest Neighbor Search:** KD-Tree for efficient similarity search.
* **Dataset:** Currently configured for the CUB-200 (Caltech-UCSD Birds 200) dataset.
* **Libraries:** Tkinter (for the GUI), OpenCV (cv2), NumPy, Matplotlib, Scikit-learn (Logistic Regression, KDTree), Time, OS, Shutil.

**Usage:**

1.  **Clone the repository.**
2.  **Ensure you have the required libraries installed:**
    ```bash
    pip install tkinter matplotlib opencv-python numpy scikit-learn
    ```
3.  **Download the pre-computed features and the CUB-200 dataset (or modify the code to use your own).** The code expects the feature files (`.npy`) and the dataset structure as defined in the script.
4.  **Run the main application script:**
    ```bash
    python your_script_name.py  # Replace 'your_script_name.py' with the actual filename
    ```
5.  **Use the GUI:**
    * Click the "Upload Bird Image" button to select an image.
    * Click "Run DCNN Algorithm & View Identified Species" to process the image and see the top K similar bird images with their predicted names.
    * Click "View Score Graph" to see a visual representation of the similarity scores.


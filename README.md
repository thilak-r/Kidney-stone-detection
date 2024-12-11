![Visited Count](https://komarev.com/ghpvc/?username=thilak-r&repo=Kidney-stone-detection&style=flat-square&label=Visited%20Count)

# Kidney Stone Detection 🩺

## **Project Overview**
This project is a deep learning-based system designed to detect kidney stones and other abnormalities from medical images (CT scans). By leveraging the power of **ResNet18** architecture, the model classifies images into four categories: **Cyst**, **Normal**, **Stone**, and **Tumor**. This system is developed to assist healthcare professionals in diagnosing kidney-related conditions with high accuracy and speed.

---

## **Key Features**
- **Deep Learning-Powered**: Utilizes **ResNet18**, a cutting-edge convolutional neural network architecture.
- **Multi-Class Classification**: Classifies medical images into four distinct categories: **Cyst**, **Normal**, **Stone**, and **Tumor**.
- **User-Friendly Interface**: Includes drag-and-drop functionality with an additional "Choose File" option for uploading images.
- **Performance Metrics**: Provides a confusion matrix, classification report, and ROC curve for thorough performance evaluation.
- **Real-Time Predictions**: Offers immediate predictions with confidence scores for uploaded images.

---

## **Technologies Used**

| **Technology**        | **Description**                                                                |
|------------------------|--------------------------------------------------------------------------------|
| **Python**            | Core programming language for model training and implementation                |
| **PyTorch**           | Deep learning framework for training and fine-tuning the ResNet18 model        |
| **Flask**             | Lightweight web framework for building the application backend                 |
| **HTML/CSS**          | Frontend for the user interface, including drag-and-drop functionality and animations |
| **Bootstrap**         | CSS framework for styling and responsive design                                |
| **Matplotlib/Seaborn**| Used for visualizing confusion matrix, ROC curve, and training performance      |

---
## **Project Structure**

```plaintext
.
├── app.py                # Backend Flask application
├── static/               # Folder containing CSS, JavaScript, and images for the frontend
├── templates/            # Folder containing HTML templates
│   └── index.html        # Main HTML file for the user interface
├── pneumonia_model.pth   # Trained PyTorch model (not included due to size limits)
├── requirements.txt      # Dependencies for the project
├── README.md             # Project documentation
└── datasets/             # Folder for training and testing datasets
```
---

## 🚀 **How to Run This Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/thilak-r/Kidney-stone-detection.git
   cd Kidney-stone-detection

2. Install Dependencies: 
Run the following command to install all required dependencies:
   ```bash 
   pip install -r requirements.txt

3. Run the Flask App
Start the Flask application by running:
   ```bash
   python app.py

4. Open Your Browser
Navigate to the following address in your web browser to access the application:
   ```bash
   http://127.0.0.1:5000
---

### 🙌 **Contributors**
#### [Thilak R](https://github.com/thilak-r) <br>
#### under guidance of [Dr Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu) <br>
---

❤️ Thank You!
Thank you for checking out our project! We hope this inspires you to explore the intersection of AI and healthcare. Feel free to reach out for questions, suggestions, or collaborations.

<br><br>

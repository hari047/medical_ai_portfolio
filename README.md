# Hari Prasad Rangaraj
### AI & Machine Learning Engineer | Medical AI Specialist
📍 Adelaide, SA | 📧 harirangaraj97@gmail.com | 🔗 https://www.linkedin.com/in/hari-rangaraj/

---

## 🚀 Profile
Master’s student in AI & Machine Learning at the University of Adelaide with a background in software engineering (Ex-Oracle). Passionate about applying **Computer Vision** and **Predictive Analytics** to solve complex problems in the MedTech space.

This repository demonstrates my ability to build AI systems from the ground up—from raw mathematical implementations of algorithms to deploying modular Deep Learning pipelines for medical imaging.

---

## 📂 Project Highlights

| Project | Domain | Tech Stack | Key Achievement |
| :--- | :--- | :--- | :--- |
| **[Blood Cell Classification](./blood-cell-classification-resnet)** | Deep Learning / MedTech | PyTorch, ResNet-18 | **95.4% Accuracy** on microscopic cell classification. |
| **[Predictive Analytics Lib](./predictive-analytics-lib)** | Algorithm Design | Python, NumPy | Built **Random Forest & K-D Tree** from scratch (No Sklearn). |
| **[Computer Vision Utils](./image-processing-utils)** | Computer Vision | NumPy, Linear Algebra | Implemented custom **Convolution & Pyramids** without OpenCV. |

---

## 🛠️ Technical Projects

### 1. 🩸 Blood Cell Classification (ResNet-18)
**Goal:** Automate the detection of blood cell subtypes from microscopic images to assist in rapid diagnosis.

* **Architecture:** Custom implementation of **ResNet-18** with a modular `Trainer` class.
* **Engineering:** Decoupled the model architecture from the training loop for reusability.
* **Performance:** Achieved **95.4% Validation Accuracy**, outperforming standard CNN baselines by leveraging residual connections to prevent vanishing gradients.

### 2. 🌲 Predictive Analytics Library (Random Forest)
**Goal:** Build a robust classification engine for tabular data (e.g., patient risk records) from first principles.

* **The "Hard" Part:** Implemented **K-D Trees** for spatial partitioning and **Bootstrap Aggregating (Bagging)** manually using NumPy.
* **Why from scratch?** To demonstrate a deep understanding of the mathematical foundations of ensemble learning, rather than relying on `sklearn.ensemble`.

**Usage:**
```python
from random_forest import RandomForest

# Initialize modular forest
model = RandomForest(n_trees=10, subsample_ratio=0.8)

# Train & Predict (Scikit-learn style API)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 3. 📷 Computer Vision Primitives
**Goal:** Develop a lightweight, foundational image processing library to pre-process raw medical scans before they enter a Neural Network.

* **The "Hard" Part:** Implemented 2D **Convolution Kernels** (Gaussian, Sobel) and **Gaussian Pyramids** using raw matrix multiplication (NumPy) rather than relying on OpenCV or PIL.
* **Why it matters:** Understanding the mathematical operations behind "filters" is crucial for debugging complex CNN architectures in medical imaging.

**Usage:**
```python
from image_ops import ImageOps

# 1. Load raw scan (normalizes to 0-1 range)
image = ImageOps.load_image("xray_sample.jpg")

# 2. Apply Custom Gaussian Blur (Noise Reduction)
kernel = ImageOps.gaussian_kernel(size=5, sigma=1.0)
clean_image = ImageOps.convolve_2d(image, kernel)

# 3. Generate Image Pyramid (Multi-scale analysis)
pyramid = ImageOps.gaussian_pyramid(clean_image, levels=4)
```

---

## 💻 Skills & Tools

| Category | Technologies |
| :--- | :--- | 
| **Deep Learning** | "PyTorch, TensorFlow, Keras, CNNs (ResNet, VGG), Transformers" |
| **Algorithms** | "Random Forest, K-D Trees, Gradient Descent, Bootstrap Aggregating" |
| **Data Science** | "NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn" |
| **Engineering** | "Python, C++, SQL, Git, Docker, CI/CD" |
| **Soft Skills** | "Technical Leadership (MiTSA), Project Management, Agile" |

---
## 📜 Certifications & Education

Master of Artificial Intelligence & Machine Learning | University of Adelaide (Adelaide University)

Bachelor of Engineering (Computer Science) | B.M.S College of Engineering

Former Software Engineer | Oracle Cerner (2020 - 2024)

---

### ⚖️ Disclaimer
*This repository contains custom implementations of machine learning algorithms developed for portfolio demonstration. While based on academic concepts, the code has been refactored and is not intended for direct use in university submissions.*

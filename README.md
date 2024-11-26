# **Bird Species Classification Using CNNs**

### **Overview**  
This project explores a convolutional neural network (CNN)-based approach for **Bird Species Classification**. The model's performance is analyzed across various optimization techniques, providing insights into its learning behavior and interpretability through **Class Activation Maps (CAM)**.  

---

### **Features**  

1. **Model Architecture**:  
   - A robust CNN architecture featuring residual connections, Batch Normalization, and Adaptive Average Pooling for efficient representation learning.  

2. **Training Insights**:  
   - Performance metrics, including **training and validation loss/accuracy trends across epochs**, are visualized for clear understanding.  

3. **Optimization Techniques**:  
   - Impact of methods such as **data augmentation**, **dropout**, **Compound Scaling Laws**, and **StepLR scheduling** on model accuracy is explored and tabulated.  

4. **Class Activation Maps (CAM)**:  
   - Visualization of regions focused on by the model for classification decisions, providing interpretability and identifying areas of misclassification.  

---

### **Model Architecture**  

| Layer                    | Configuration                        | Output Shape         |
|--------------------------|---------------------------------------|----------------------|
| Input                    | Image (300x300x3)                    | (300, 300, 3)        |
| Conv2D + BatchNorm + ReLU| 7x7, stride 2, padding 3             | (64, 150, 150)       |
| MaxPool2D               | 3x3, stride 2                        | (64, 75, 75)         |
| Residual Blocks          | Several Conv2D layers with skip connections and strides| Intermediate (128 to 512 channels) |
| AdaptiveAvgPool2D        | Adaptive Average Pooling             | (512, 1, 1)          |
| Fully Connected Layer    | Linear (512 → 10)                    | (10)                 |

The table captures the architecture of the **birdClassifier** CNN, optimized for scalability and feature extraction.  

---

### **Performance Metrics**  

#### **Training and Validation Trends**  
Plots for **Loss** and **Accuracy** across epochs highlight model convergence and generalization:  
- Training-to-validation split: **80% train**, **20% validation**.  

#### **Optimization Techniques**  
| Technique               | Validation Accuracy | Notes                                          |
|--------------------------|---------------------|------------------------------------------------|
| No Optimization          | 60.13%             | Baseline model                                |
| Compound Scaling Laws    | 81.78%             | Improved scaling                              |
| Data Augmentation        | 89.98%             | Enhanced generalization with augmented data   |
| Dropout                  | 86.50%             | Slower learning but robust performance        |
| StepLR (all combined)    | **90.72%**         | Best accuracy achieved                        |

---

### **Class Activation Maps (CAM)**  

**CAM visualizations** provide insights into the model's decision-making:  
- Heatmaps reveal the image regions most influential in classification.  
- Misclassified examples show scattered attention, indicating potential areas for further optimization.  

#### **Examples of CAM**:  
- Correct classifications emphasize relevant textures and objects in the images.  
- Misclassified examples display poor focus, leading to incorrect predictions.  

---

### **Usage**  

1. Clone the repository:  
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Running the Code
   ```bash
   # Running code for training. save the model in the same directory with name "bird.pth"
   python bird.py path_to_dataset train bird.pth 
  
   # Running code for inference
   python bird.py path_to_dataset test bird.pth
   ```

---

### **Authors**  
- **Rishit Jakharia**  
- **Aditya Jha** 

---

### **Acknowledgments**  
This work was completed as part of **COL333 Assignment 3.1** at **IIT Delhi**, exploring advanced deep learning techniques under academic supervision.  

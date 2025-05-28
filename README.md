# Image Classification with Deep Learning

A comprehensive computer vision project implementing and comparing different CNN architectures for multi-class image classification using TensorFlow/Keras.

## 📊 Dataset Overview

The dataset contains **13,992 images** across **6 classes**:
- **Buildings**: 2,184 images
- **Forest**: 2,264 images
- **Glacier**: 2,397 images
- **Mountain**: 2,505 images
- **Sea**: 2,267 images
- **Street**: 2,375 images

The dataset is well-balanced with relatively equal distribution across all classes.

## 🏗️ Model Architectures

### Baseline CNN Model
- **3 Convolutional Blocks**: Conv2D → MaxPooling2D
  - Block 1: 32 filters (3×3), ReLU activation
  - Block 2: 64 filters (3×3), ReLU activation
  - Block 3: 128 filters (3×3), ReLU activation
- **Fully Connected Layers**: 
  - Dense(512) → Dense(256) → Dense(128)
  - Output: Dense(6) with softmax activation
- **Total Parameters**: ~16.8M

### Deeper Model
Enhanced architecture with additional layers and improved performance capabilities.

## 📈 Results Comparison

| Model | Optimizer | Test Accuracy | Performance |
|-------|-----------|---------------|-------------|
| Baseline | Adam | 72.07% | ✅ Good baseline |
| Deeper | Adam | 76.63% | ⬆️ +4.56% improvement |
| Deeper | SGD | **77.23%** | 🏆 **Best performance** |

## 🎯 Model Performance Analysis

### Baseline Model (Adam Optimizer)
- **Test Accuracy**: 72.07%
- **Test Loss**: 0.7627
- **Training Time**: 94/94 steps per epoch

**Classification Report**:
```
Class         Precision  Recall  F1-Score  Support
buildings     0.52       0.76    0.62      437
forest        0.85       0.95    0.90      474
glacier       0.78       0.66    0.71      553
mountain      0.57       0.57    0.57      525
sea           0.68       0.74    0.71      510
street        0.79       0.68    0.73      501

Macro Avg     0.73       0.73    0.72      3000
Weighted Avg  0.74       0.72    0.72      3000
```

### Deeper Model (SGD Optimizer)
- **Test Accuracy**: 77.23% ⭐
- **Best Validation Accuracy**: 77.59% at epoch 18
- **Training**: 511s per step

**Classification Report**:
```
Class         Precision  Recall  F1-Score  Support
buildings     0.56       0.84    0.67      437
forest        0.92       0.94    0.93      474
glacier       0.75       0.69    0.72      553
mountain      0.81       0.66    0.72      525
sea           0.77       0.78    0.77      510
street        0.84       0.69    0.76      501

Macro Avg     0.77       0.77    0.76      3000
Weighted Avg  0.78       0.76    0.76      3000
```

## 🔧 Training Configuration

### SGD Optimizer Setup
```python
# SGD with momentum and learning rate scheduling
sgd_optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01, 
    momentum=0.9
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,          # Reduce LR by 20%
    patience=2,          # Wait 2 epochs
    min_lr=1e-6,         # Minimum LR threshold
    verbose=1
)
```

## 📊 Key Insights

1. **Optimizer Impact**: SGD with momentum outperformed Adam optimizer by **0.6%**
2. **Architecture Depth**: Deeper models showed **4.56%** improvement over baseline
3. **Class Performance**: 
   - **Best performing**: Forest class (94% recall)
   - **Most challenging**: Mountain class (66% recall)
4. **Training Stability**: Learning rate scheduling and early stopping prevented overfitting

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow
pip install matplotlib
pip install numpy
pip install scikit-learn
```

### Running the Code
```python
# Clone the repository
git clone https://github.com/beater35/deep-cnn-image-classification.git
cd  deep-cnn-image-classification

# Load and run the Jupyter notebook
jupyter notebook classification_model_comparison_wine_quality.ipynb
```

### Usage
```python
# Load and preprocess data
train_dir = "/content/drive/MyDrive/ai_ml/train"
test_dir = "/content/drive/MyDrive/ai_ml/test"

# Create model
model = create_baseline_model(input_shape, num_classes)

# Train with SGD optimizer
model.compile(
    optimizer=sgd_optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fit model with callbacks
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)
```

## 📁 Project Structure
```
├── classification_model_comparison_image_classification.ipynb
└── README.md
```

## 📬 Contact

If you're interested in accessing the dataset or have any questions about this project, feel free to reach out:

  - Email: bideets3035@gmail.com

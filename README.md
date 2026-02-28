# Network Configuration Document  

## 1. Architecture Description  

For this project, we implemented a fully connected **Multi-Layer Perceptron (MLP)** to classify grayscale landscape images into 6 categories.

The input tensor has shape `[batch_size, 1, 150, 150]`. Since the model is fully connected, the first step is to flatten the image. The **1 × 150 × 150 = 22500** pixels are converted into a vector of size 22500.

After flattening, the network consists of three linear layers with decreasing dimensionality:

- 22500 → 1024  
- 1024 → 512  
- 512 → 256  

After each linear layer, we applied:

- Batch Normalization  
- LeakyReLU activation  
- Dropout  

The final layer is `Linear(256 → 6)`, producing one logit per class. We did not include a softmax layer explicitly, since `CrossEntropyLoss` already applies it internally.

The progressive reduction in dimensionality (1024 → 512 → 256) compresses the learned representation before the final classification step.

---

## 2. Input Size and Preprocessing  

All images were converted to grayscale. This reduces the number of parameters in the first linear layer, which is the largest component of the network. Using RGB would have tripled the number of input features and increased both computational cost and overfitting risk.

The final input size is **150 × 150 pixels, single channel**.

### Training preprocessing  

For the training set, we applied:

- Grayscale conversion  
- Resize to 150×150  
- RandomHorizontalFlip (p=0.5)  
- RandomRotation (±8 degrees)  
- Brightness jitter (0.2)  
- ToTensor  
- Normalize(mean=[0.5], std=[0.5])  

The random transformations were applied only during training to increase variability and reduce overfitting.

### Validation and test preprocessing  

For validation, internal test, and competition test sets, we applied:

- Grayscale  
- Resize  
- ToTensor  
- Normalize  

No data augmentation was used at evaluation time. All evaluation images go through the same deterministic preprocessing pipeline so that validation accuracy reflects inference behavior.

---

## 3. Loss Function  

We used `CrossEntropyLoss(label_smoothing=0.05)`.

This is a standard loss for multi-class classification. Label smoothing distributes a small portion of the probability mass across non-target classes, which reduces overconfident predictions and improves stability during training.

---

## 4. Optimizer and Hyperparameters  

- **Optimizer:** Adam  
- **Initial learning rate:** 0.0005  
- **Weight decay:** 1e-4  
- **Batch size:** 32  
- **Maximum epochs:** 100  

We used `ReduceLROnPlateau`, monitoring validation accuracy. If accuracy does not improve for 7 epochs, the learning rate is reduced by a factor of 0.5. The minimum learning rate is set to 1e-6.

This helps when training reaches a plateau.

---

## 5. Regularization  

To control overfitting, we applied:

- Data augmentation (training only)  
- Dropout: 0.3, 0.4, and 0.3 in hidden layers  
- Batch Normalization after each hidden layer  
- Weight decay  
- Label smoothing  
- Learning rate scheduling  
- Early stopping based on validation accuracy  

Whenever validation accuracy improves, the model is saved to `best_model.pth`. If no improvement is observed within the patience window, training stops. At the end, the best-performing model is reloaded.

---

## 6. Model Selection Strategy  

From `data/seg_test`, we created two subsets using `StratifiedShuffleSplit`:

- **Validation set:** 100 images  
- **Internal test set:** 100 images  

Stratification preserves the original class distribution.

After each epoch, validation accuracy is computed and tracked:

- `best_val_acc`  
- `best_epoch`  

If accuracy improves, the model state is saved and the patience counter is reset.

The final model corresponds to the highest validation accuracy achieved during training.

---

## Resultados de diferentes arquitecturas

| Archivo | Model Type | # Layers | Optim. Method | Learning Rate | # Epochs | Activation function | Regul. | Val. Split | Training Accuracy | Test Accuracy |
|---|---|---:|---|---:|---:|---|---|---|---:|---:|
| modelo1, rama 1 | MLP | 3 | Adam | 0.0005 | 55 | LeakyReLU | BatchNorm + Dropout + L2 + label smoothing + scheduler + early stopping | Estratificado (2 muestras de 100) | 59.33% | 64.00% |
| modelo2, rama 2 | MLP | 3 | Adam | 0.0005 | 91 | LeakyReLU | BatchNorm + Dropout + L2 + label smoothing + scheduler + early stopping | Estratificado (test_size=0.3) | 56.32% | 58.89% |
| modelo3, rama 3  | MLP | 4 | Adam | 0.0005 | 90 | LeakyReLU | BatchNorm + Dropout + L2 + scheduler + early stopping | Estratificado (test_size=0.3) | 88.79% | 62.56% |
| modelo4, rama 4  | CNN | 3 | Adam | 0.0001 | 75 | ReLU | BatchNorm + Dropout + L2 | Estratificado (2 muestras de 100) | 61.70% | 64.00% |


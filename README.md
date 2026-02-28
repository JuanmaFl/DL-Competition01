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

## Results  

| MODELS | LAYERS                              | OPTIM METHOD              | LEARNING RATE | ACTIVATION FUNCTION        | REGULATION                                                                 | VAL. SPLIT                         | # EPOCH              | TRAIN ACCURACY | VAL. ACCURACY |
|--------|--------------------------------------|---------------------------|---------------|----------------------------|---------------------------------------------------------------------------|------------------------------------|----------------------|----------------|----------------|
| NB4    | 3 (1024 → 512 → 256 → 6)            | Adam                      | 0.0001        | ReLU                       | Dropout (0.3 / 0.4 / 0.3) <br> BatchNorm <br> weight_decay 1e-4        | 100 samples (stratified)           | 75                   | 61.70%         | 67.00%        |
| NB1    | 3 (1024 → 512 → 256 → 6)            | Adam + ReduceLROnPlateau | 0.0005        | LeakyReLU (0.15 / 0.1)     | Dropout (0.3 / 0.4 / 0.3) <br> BatchNorm <br> LabelSmooth 0.05 <br> weight_decay 1e-4 | 100 samples (stratified) | 100 (best ep. 55) | 62.33%         | 64.00%        |
| NB2    | 3 (1024 → 512 → 256 → 6)            | Adam + ReduceLROnPlateau | 0.0005        | LeakyReLU (0.1)            | Dropout (0.5 / 0.45 / 0.35) <br> BatchNorm <br> LabelSmooth 0.1 <br> weight_decay 1e-3 | 70% val / 30% test (stratified) | 100 (best ep. 91) | 58.11%         | 57.10%        |
| NB3    | 4 (2048 → 1024 → 512 → 256 → 6)     | Adam + ReduceLROnPlateau | 0.0005        | LeakyReLU (0.1)            | Dropout (0.25 / 0.25 / 0.2 / 0.15) <br> BatchNorm <br> weight_decay 5e-5 | 70% val / 30% test (stratified) | 100 (best ep. 90) | 89.96%         | 60.05%        |


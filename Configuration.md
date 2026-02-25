# Network Configuration Document

## 1. Architecture Description

We use a fully connected Multi-Layer Perceptron (MLP) for grayscale landscape image classification into 6 classes.

- **Input tensor**: shape `[batch_size, 1, 150, 150]`
- **Layer stack**:
  1. `Flatten` (from \(1 \times 150 \times 150 = 22500\) pixels to a 22500‑dimensional vector)
  2. `Linear(22500 → 1024)`  
     `BatchNorm1d(1024)`  
     `LeakyReLU(negative_slope=0.15)`  
     `Dropout(p=0.3)`
  3. `Linear(1024 → 512)`  
     `BatchNorm1d(512)`  
     `LeakyReLU(negative_slope=0.1)`  
     `Dropout(p=0.4)`
  4. `Linear(512 → 256)`  
     `BatchNorm1d(256)`  
     `LeakyReLU(negative_slope=0.1)`  
     `Dropout(p=0.3)`
  5. **Output layer**: `Linear(256 → 6)` (one logit per class)

The final softmax is implicit in the loss function (`CrossEntropyLoss`), which operates on logits.

---

## 2. Input Size and Preprocessing

- **Input size**: single‑channel grayscale images of **150 × 150** pixels.

### Training preprocessing (`train_transform`)

Applied to images from `data/seg_train`:

1. `Grayscale(num_output_channels=1)` – convert to 1‑channel grayscale.
2. `Resize((150, 150))` – resize to fixed 150×150.
3. `RandomHorizontalFlip(p=0.5)` – horizontal flip with probability 0.5.
4. `RandomRotation(8)` – random rotation up to ±8 degrees.
5. `ColorJitter(brightness=0.2)` – random brightness perturbation.
6. `ToTensor()` – convert to PyTorch tensor, pixel values in \([0, 1]\).
7. `Normalize(mean=[0.5], std=[0.5])` – normalize to roughly \([-1, 1]\).

### Validation / Internal Test / Competition Test preprocessing (`test_transform`)

Applied to:
- Validation and internal test splits from `data/seg_test`.
- Competition images from `data/seg_pred` (comp_test subset used in the notebook).

Pipeline:

1. `Grayscale(num_output_channels=1)`
2. `Resize((150, 150))`
3. `ToTensor()`
4. `Normalize(mean=[0.5], std=[0.5])`

No data augmentation is used at evaluation time.

---

## 3. Loss Function

- **Loss**: `nn.CrossEntropyLoss(label_smoothing=0.05)`
  - Multiclass cross‑entropy over 6 classes.
  - `label_smoothing=0.05` reduces over‑confidence by distributing a small portion of probability mass to non‑target classes.

---

## 4. Optimizer and Hyperparameters

- **Optimizer**: `torch.optim.Adam`
  - **Learning rate**: `0.0005`
  - **Weight decay**: `1e-4` (L2 regularization on weights)

- **Batch size**:
  - Training, validation and internal test: `batch_size = 32`
  - Competition test loader: `batch_size = 32` (inference only)

- **Number of epochs**:
  - Maximum of `100` epochs.
  - Effective number of epochs can be lower due to early stopping (see Regularization).

- **Scheduler (used)**:
  - `torch.optim.lr_scheduler.ReduceLROnPlateau`
    - `mode='max'` (monitors validation accuracy)
    - `factor=0.5` (halves the learning rate when there is no improvement)
    - `patience=7` epochs without improvement before reducing LR
    - `min_lr=1e-6` (lower bound for the learning rate)

---

## 5. Regularization Methods

We apply several regularization techniques:

- **Data augmentation (training only)**:
  - Horizontal flips (`RandomHorizontalFlip(p=0.5)`)
  - Small random rotations (`RandomRotation(8)`)
  - Brightness perturbations (`ColorJitter(brightness=0.2)`)

- **Dropout**:
  - After first hidden layer: `Dropout(p=0.3)`
  - After second hidden layer: `Dropout(p=0.4)`
  - After third hidden layer: `Dropout(p=0.3)`

- **Batch Normalization**:
  - Applied after each hidden linear layer (1024, 512, 256 units) to stabilize training.

- **Weight decay**:
  - `weight_decay = 1e-4` in the Adam optimizer.

- **Label smoothing**:
  - `label_smoothing = 0.05` in `CrossEntropyLoss` to avoid overly confident predictions.

- **Learning rate scheduling**:
  - `ReduceLROnPlateau` lowers the learning rate when validation accuracy plateaus, acting as an additional regularization and convergence aid.

- **Early stopping**:
  - Training monitors validation accuracy.
  - If validation accuracy does not improve for a fixed patience window, training is stopped early.
  - The best model (highest validation accuracy) is saved to `best_model.pth` and reloaded after training.

---

## 6. Model Selection Strategy

- **Validation/Test split**:
  - Starting from `data/seg_test`, we build two stratified subsets using `StratifiedShuffleSplit`:
    - **Validation set**: 100 images.
    - **Internal test set**: 100 images.
  - Stratification guarantees that both subsets preserve the original class distribution.

- **Selection criterion**:
  - After each epoch, we compute validation accuracy on the validation set.
  - We keep track of:
    - `best_val_acc`: best validation accuracy observed so far.
    - `best_epoch`: epoch where `best_val_acc` is achieved.
  - Whenever validation accuracy improves:
    - The model’s `state_dict` is saved to `best_model.pth`.
    - The early‑stopping patience counter is reset.

- **Final model**:
  - After training stops (due to max epochs or early stopping), we reload `best_model.pth`:
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    
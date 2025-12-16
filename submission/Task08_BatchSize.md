# Task 08 — Batch Size

## 1. Objective
Compare model performance with different batch sizes (8, 32, 128) to understand how batch size affects training stability, convergence speed, and generalization.

## 2. Code Used
```python
batch_sizes = [8, 32, 128]

for bs in batch_sizes:
    model = models.clone_model(base_model)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=bs,
        validation_data=(x_val, y_val),
        verbose=0
    )
```

## 3. Results
Validation loss curves for batch sizes 8, 32, and 128 are plotted and saved to `results/loss_curves/batchsize_comparison.png`. The curves show how batch size affects training dynamics and final performance.

## 4. Short Analysis
Batch size significantly impacts **optimizer behavior** and model **generalization**. Smaller batches (8) provide more frequent weight updates with higher variance gradients, which can help escape local minima but may lead to noisier training. Medium batches (32) balance stability and update frequency. Larger batches (128) provide smoother gradients and faster training per epoch but fewer updates per epoch, potentially requiring more epochs to converge. 
The **Adam optimizer** adapts to the gradient variance introduced by different batch sizes. Smaller batches often show better **generalization** due to the implicit regularization effect of noisy gradients, while larger batches may converge faster but risk overfitting. 
The **ReLU activation** gradients are averaged over the batch, so batch size affects the smoothness of these gradients.

## 5. Key Takeaway
Smaller batch sizes provide implicit regularization through noisy gradients, improving generalization, while larger batches offer faster training per epoch but may require more epochs to converge.


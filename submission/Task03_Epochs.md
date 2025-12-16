# Task 03 — Epochs

## 1. Objective
Compare model performance across different numbers of training epochs (5, 10, 20) to understand the relationship between training duration and validation loss.

## 2. Code Used
```python
epoch_settings = [5, 10, 20]
epoch_histories = {}

for e in epoch_settings:
    model = models.clone_model(base_model)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    epoch_histories[e] = model.fit(
        x_train, y_train,
        epochs=e,
        validation_data=(x_val, y_val),
        verbose=0
    )
```

## 3. Results
Validation loss curves for 5, 10, and 20 epochs are plotted and saved to `results/loss_curves/epoch_comparison.png`. The curves show how validation loss decreases with more training epochs.

## 4. Short Analysis
More epochs generally lead to lower validation loss, as the **Adam optimizer** continues to update weights and reduce the loss function. However, there's a point of diminishing returns—after a certain number of epochs, the model may start to **overfit** (training loss continues decreasing while validation loss plateaus or increases). The comparison reveals the trade-off between training time and model performance. The **optimizer behavior** (Adam's adaptive learning rates) helps the model converge efficiently across different epoch counts.

## 5. Key Takeaway
More epochs improve performance up to a point, but excessive training can lead to overfitting; finding the optimal epoch count balances performance and generalization.


# Task 04 — Early Stopping

## 1. Objective
Implement early stopping to automatically halt training when validation loss stops improving, preventing overfitting and saving computational resources.

## 2. Code Used
```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model_es = models.clone_model(base_model)
model_es.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history_es = model_es.fit(
    x_train, y_train,
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=[early_stop]
)
```

## 3. Results
Training stopped automatically after 10 epochs (out of 30 maximum) when validation loss stopped improving for 3 consecutive epochs. The model restored the best weights from epoch 7 (val_loss: 0.0783).

## 4. Short Analysis
Early stopping is a form of **regularization** that prevents **overfitting** by monitoring validation loss. When validation loss stops decreasing for `patience` epochs, training halts. The `restore_best_weights=True` parameter ensures the model uses the weights from the epoch with the lowest validation loss, not the final epoch. This technique helps maintain **generalization** by stopping before the model memorizes training data. The **optimizer behavior** (Adam) continues to reduce training loss, but early stopping prevents the model from overfitting to training patterns that don't generalize.

## 5. Key Takeaway
Early stopping automatically prevents overfitting by halting training when validation performance plateaus, ensuring the model maintains good generalization without manual intervention.


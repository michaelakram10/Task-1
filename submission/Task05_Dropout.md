# Task 05 — Dropout

## 1. Objective
Compare model performance with different dropout rates (0.0, 0.1, 0.3) to understand how dropout regularization affects validation loss and prevents overfitting.

## 2. Code Used
```python
dropout_rates = [0.0, 0.1, 0.3]

for d in dropout_rates:
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(d),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val), verbose=0)
```

## 3. Results
Validation loss curves for dropout rates 0.0, 0.1, and 0.3 are plotted and saved to `results/loss_curves/dropout_comparison.png`. The curves show how different dropout rates affect model generalization.

## 4. Short Analysis
Dropout is a **regularization** technique that randomly sets a fraction of neurons to zero during training, forcing the network to learn redundant representations and preventing **overfitting**. A dropout rate of 0.0 (no dropout) may show lower training loss but higher validation loss due to overfitting. Moderate dropout (0.1) can improve **generalization** by reducing reliance on specific neurons. Higher dropout (0.3) may underfit if too many neurons are disabled. The **ReLU activation** combined with dropout helps the model learn more robust features. Dropout affects the **optimizer behavior** by introducing noise during training, which regularizes the weight updates.

## 5. Key Takeaway
Dropout regularization prevents overfitting by randomly disabling neurons during training, improving generalization, but excessive dropout can lead to underfitting.

---

## 🗣️ شرح بالعامية

### إيه هو الـ Dropout؟
تخيل إنك في فريق شغل، وكل يوم بتطفي راندوم شوية موظفين. الباقيين لازم يتعلموا يشتغلوا من غيرهم!

### إزاي بيشتغل؟
- في كل مرة تدريب، بنطفي نسبة معينة من الـ neurons عشوائياً
- Dropout 0.1 = بنطفي 10% من الـ neurons
- Dropout 0.3 = بنطفي 30% من الـ neurons

### ليه ده مفيد؟
- الموديل مبيعتمدش على neurons معينة
- بيتعلم يكون أقوى وأذكى
- بيمنع الـ overfitting

### إيه اللي جربناه؟
- 0.0 (مفيش dropout) - ممكن يحصل overfitting
- 0.1 (10%) - توازن كويس
- 0.3 (30%) - ممكن يبقى كتير قوي

### الخلاصة
الـ Dropout زي التدريب الصعب - بيخلي الموديل أقوى، بس لو زودناه قوي الموديل مش هيتعلم حاجة! ⚖️

# Task 06 — L2 Regularization


##  شرح

### إيه هو الـ L2 Regularization؟
تقليل ال wieghts بتاعة الموديل بنائا علي كبرها

### إزاي بيشتغل؟
- بنضيف عقوبة على الـ loss بناءً على حجم الـ weights
- كل ما الـ weights أكبر، العقوبة أكبر
- الموديل بيحاول يخلي الـ weights صغيرة

### ليه ده مفيد؟
- الـ weights الكبيرة بتخلي الموديل "يحفظ" التدريب
- لما نخليها صغيرة، الموديل بيتعلم patterns عامة
- بيحسن الـ generalization

### إيه اللي جربناه؟
- 0.0001 - عقوبة خفيفة
- 0.001 - عقوبة متوسطة
- 0.01 - عقوبة قوية

### الفرق بين L2 و Dropout؟
- **Dropout**: بيطفي neurons عشوائياً
- **L2**: بيخلي الـ weights صغيرة

---


## 1. Objective
Compare model performance with different L2 regularization strengths (0.0001, 0.001, 0.01) to understand how weight penalty affects model generalization.

## 2. Code Used
```python
l2_values = [0.0001, 0.001, 0.01]

for l2v in l2_values:
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2v)),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val), verbose=0)
```

## 3. Results
Models were trained with L2 regularization values of 0.0001, 0.001, and 0.01. The validation loss and accuracy were recorded for each configuration to compare the effect of different regularization strengths.

## 4. Short Analysis
L2 regularization adds a penalty term to the loss function proportional to the sum of squared weights, encouraging smaller weights and preventing **overfitting**. A small L2 value (0.0001) provides mild regularization, while larger values (0.001, 0.01) impose stronger constraints. The **optimizer behavior** (Adam) adjusts weight updates to balance the data loss and the regularization penalty. Too much L2 regularization can cause **underfitting** by constraining the model too much, while too little may not prevent overfitting. The **ReLU activation** combined with L2 regularization helps maintain sparse, efficient weight patterns. This technique improves **generalization** by reducing model complexity.

## 5. Key Takeaway
L2 regularization prevents overfitting by penalizing large weights, but the regularization strength must be carefully tuned—too weak allows overfitting, too strong causes underfitting.


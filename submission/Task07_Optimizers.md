# Task 07 — Optimizers

## 1. Objective
Compare the performance of different optimization algorithms (SGD, Momentum, Adam, AdamW) to understand how optimizer choice affects training dynamics and convergence.

## 2. Code Used
```python
optimizers = {
    "SGD": tf.keras.optimizers.SGD(0.01),
    "Momentum": tf.keras.optimizers.SGD(0.01, momentum=0.9),
    "Adam": tf.keras.optimizers.Adam(),
    "AdamW": tf.keras.optimizers.AdamW()
}

for name, opt in optimizers.items():
    model = models.clone_model(base_model)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=0)
```

## 3. Results
Validation loss curves for SGD, Momentum, Adam, and AdamW optimizers are plotted and saved to `results/optimizer_tests/optimizer_comparison.png`. The curves show convergence speed and final performance for each optimizer.

## 4. Short Analysis
Different optimizers have distinct **optimizer behavior** characteristics. 
**SGD** uses fixed learning rates and may converge slowly. 
**Momentum** adds velocity to gradient updates, helping escape local minima and converge faster. 
**Adam** adapts learning rates per parameter using moving averages of gradients and squared gradients, typically converging faster and more reliably. 
**AdamW** decouples weight decay from gradient updates, improving generalization compared to Adam. The choice of optimizer affects how the **ReLU activation** gradients are processed and how weights are updated. Adam and AdamW generally show better **generalization** and faster convergence on this task, while SGD may require more epochs to reach similar performance.

## 5. Key Takeaway
Adam and AdamW optimizers typically converge faster and achieve better performance than SGD, with adaptive learning rates that adjust per parameter during training.

---

## 🗣️ شرح بالعامية

### إيه هو الـ Optimizer؟
الـ Optimizer هو اللي بيقرر إزاي الموديل يتعلم - يعني إزاي يعدّل الـ weights بتاعته.

### الـ Optimizers اللي جربناها:

**1. SGD (Stochastic Gradient Descent)**
- زي واحد بيمشي بخطوات ثابتة
- بطيء بس بسيط
- ممكن يتوه في المنحدرات

**2. Momentum**
- زي كورة بتتدحرج - بتاخد سرعة مع الوقت
- أسرع من SGD
- بيعدي المطبات الصغيرة

**3. Adam**
- أذكى optimizer - بيظبط السرعة لكل weight لوحده
- سريع ومستقر
- الأشهر والأكتر استخداماً

**4. AdamW**
- زي Adam بس بيعمل weight decay أحسن
- بيحسن الـ generalization

### النتيجة؟
Adam و AdamW كانوا الأسرع والأحسن! 🏆

### الخلاصة
الـ Optimizer زي السواق - في سواق بطيء وحذر (SGD)، وفي سواق سريع وشاطر (Adam). اختار اللي يناسب مشروعك! 🚗

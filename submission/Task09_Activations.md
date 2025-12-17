# Task 09 — Activations

## 1. Objective
Compare different activation functions (tanh, softsign, GELU) to understand how activation choice affects model performance and training dynamics.

## 2. Code Used
```python
activations = ["tanh", "softsign", tf.keras.activations.gelu]

for act in activations:
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation=act),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=0)
```

## 3. Results
Models were trained with tanh, softsign, and GELU activation functions. Training and validation metrics were recorded for each activation function to compare their performance.

## 4. Short Analysis
Different **activations** introduce distinct non-linearities that affect gradient flow and model capacity. 
**tanh** outputs values in [-1, 1] and can suffer from vanishing gradients in deep networks. 
**softsign** is similar to tanh but smoother, potentially providing better gradient flow. 
**GELU** (Gaussian Error Linear Unit) is a smooth, non-monotonic activation that often performs well in modern architectures. The choice of activation affects how the **optimizer behavior** (Adam) processes gradients—some activations provide smoother gradients than others. The **ReLU activation** (used in the base model) is simple and effective, but alternatives like GELU can sometimes improve **generalization** by introducing different non-linear patterns. Activation functions directly impact the model's ability to learn complex patterns and avoid **overfitting**.

## 5. Key Takeaway
Activation functions shape the model's non-linearity and gradient flow; GELU and other modern activations can sometimes outperform traditional choices like ReLU or tanh.

---

## 🗣️ شرح بالعامية

### إيه هي الـ Activation Function؟
الـ Activation هي اللي بتقرر الـ neuron "يشتغل" ولا "يفصل". من غيرها الموديل هيبقى خط مستقيم بسيط!

### الـ Activations اللي جربناها:

**1. tanh**
- بيطلع قيم من -1 لـ 1
- قديم ومعروف
- ❌ ممكن يعمل vanishing gradient (الموديل يتوه)

**2. softsign**
- شبه tanh بس أنعم
- بيطلع قيم من -1 لـ 1
- ✅ gradient أحسن شوية

**3. GELU**
- الجديد والفخم 🌟
- بيستخدم في GPT و BERT
- ✅ smooth ومرن
- ✅ أداء ممتاز

### إيه الفرق؟
تخيل إنك بتسأل الـ neuron "تشتغل؟":
- **ReLU**: "أيوه" أو "لأ" (0 أو الرقم)
- **tanh**: "أيوه شوية" أو "لأ شوية" (-1 لـ 1)
- **GELU**: "أيوه بنسبة كذا%" (أذكى وأنعم)

### الخلاصة
الـ Activation Function زي شخصية الـ neuron - في الحاد (ReLU)، والمرن (GELU). اختار اللي يناسب مشكلتك! 🎭

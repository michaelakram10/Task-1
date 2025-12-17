# Task 01 — Prediction Analysis
##  شرح

### إيه اللي عملناه؟
احنا جبنا 3 صور أرقام من الـ test set (أرقام 4 و 4 و 6) وطلبنا من الموديل يتوقع هما إيه.

### إزاي الكود شغال؟
- بناخد الصورة ونحطها في الموديل
- الموديل بيطلعلنا احتمالات لكل رقم من 0 لـ 9
- بنختار الرقم اللي احتماله أعلى (ده اللي الموديل متأكد منه أكتر)

### النتيجة؟
الموديل جاب الـ 3 صور صح! 

### الخلاصة
حتى موديل بسيط (طبقتين بس) يقدر يعرف الأرقام المكتوبة بخط اليد كويس جداً. الـ MNIST مش صعب قوي لأن الأرقام واضحة ومنظمة.








## 1. Objective
Analyze the base model's predictions on three test samples to verify correct digit classification.

## 2. Code Used
```python
indices = [4, 27, 88]

for i in indices:
    sample = x_test[i].reshape(1, 28, 28)
    probs = base_model.predict(sample)
    pred = np.argmax(probs)
    
    print(f"Index {i} | Predicted: {pred} | True: {y_test[i]}")
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Pred {pred} / True {y_test[i]}")
    plt.savefig(f"results/predictions/sample_{i}.png")
```

## 3. Results
- Index 4: Predicted 4, True 4 ✓
- Index 27: Predicted 4, True 4 ✓
- Index 88: Predicted 6, True 6 ✓

All predictions were correct. Sample images saved to `results/predictions/`.

## 4. Short Analysis
The base model with a simple architecture (Flatten → Dense(96, ReLU) → Dense(10, Softmax)) successfully classified all three test samples. The **softmax activation** in the output layer provides probability distributions over the 10 digit classes, and the model correctly identified digits 4, 4, and 6. This demonstrates that even a minimal neural network can achieve good performance on MNIST, which is a relatively simple classification task with clear visual patterns.

## 5. Key Takeaway
A simple two-layer neural network with ReLU activation and softmax output can effectively classify MNIST digits, achieving 100% accuracy on the tested samples.

---.

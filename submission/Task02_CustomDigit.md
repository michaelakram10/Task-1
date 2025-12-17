# Task 02 — Custom Digit

## 1. Objective
Test the base model's ability to classify a custom handwritten digit image not from the MNIST dataset.

## 2. Code Used
```python
if os.path.exists("custom_digit.png"):
    img = Image.open("custom_digit.png").convert("L")
    img = img.resize((28, 28))
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.reshape(1, 28, 28)
    
    pred = base_model.predict(img_arr)
    print("Custom Image Prediction:", pred.argmax())
```

## 3. Results
The Custom image number where : **3**
Custom Image Prediction: **5**

The model predicted the custom digit as class 5 "Which is wrong". The processed image is saved to `results/predictions/custom_digit.png`.

## 4. Short Analysis
The model successfully processed an external image by converting it to grayscale, resizing to 28x28 pixels, and normalizing pixel values to [0, 1]. The **softmax activation** produced a probability distribution, and the model classified the digit as 5. This demonstrates the model's **generalization** capability—it can handle images outside the training set, though performance depends on how similar the custom digit's style is to MNIST's training data. The preprocessing steps (grayscale conversion, resizing, normalization) are crucial for matching the model's expected input format.

## 5. Key Takeaway
Proper preprocessing (resize, grayscale, normalization) enables the model to classify custom images, demonstrating generalization beyond the training dataset.

---

## 🗣️ شرح بالعامية

### إيه اللي عملناه؟
جربنا الموديل على صورة رقم رسمناها بنفسنا (مش من الداتا الأصلية).

### إزاي جهزنا الصورة؟
1. **حولناها لأبيض وأسود** - لأن الموديل اتدرب على صور رمادي
2. **صغرناها لـ 28x28 بكسل** - نفس حجم صور التدريب
3. **قسمنا على 255** - عشان القيم تبقى بين 0 و 1

### النتيجة؟
الصورة كانت رقم 3، بس الموديل توقع إنها 5 ❌

### ليه غلط؟
لأن خط إيدنا مختلف عن خط الناس اللي كتبوا الـ MNIST. الموديل بيشتغل كويس بس لما الصورة الجديدة تكون شبه صور التدريب.

### الخلاصة
التحضير الصح للصورة مهم جداً، بس برضو الموديل ممكن يغلط لو الصورة مختلفة كتير عن اللي اتدرب عليه.

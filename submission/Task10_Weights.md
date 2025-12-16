# Task 10 — Weights

## 1. Objective
Inspect the weight matrix of the base model's first dense layer to understand the model's internal structure and parameter count.

## 2. Code Used
```python
weights, bias = base_model.layers[1].get_weights()
print("Weight matrix shape:", weights.shape)
```

## 3. Results
Weight matrix shape: **(784, 96)**

The first dense layer has 784 input features (28×28 flattened image) and 96 output neurons, resulting in 75,264 weight parameters plus 96 bias parameters.

## 4. Short Analysis
The weight matrix shape (784, 96) reveals the model's architecture: 784 inputs (flattened 28×28 MNIST images) connect to 96 hidden neurons. Each weight represents a learned connection strength between an input pixel and a hidden neuron. The **ReLU activation** applied to these weighted sums introduces non-linearity. The **optimizer behavior** (Adam) updates these weights during training to minimize the loss function. The total parameter count (75,264 weights + 96 biases = 75,360 parameters) determines model capacity—too few parameters may underfit, while too many risk **overfitting**. Understanding weight dimensions helps diagnose model complexity and **generalization** potential. The weights encode the learned patterns that enable digit classification.

## 5. Key Takeaway
Inspecting weight matrices reveals model architecture and parameter count, helping understand model capacity and potential for overfitting or underfitting.


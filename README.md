# 🧠 Adversarial AI  
### Black-Box Attacks, Decision Boundary Leakage, and Defensive Tradeoffs

![Adversarial AI banner](images/banner.png)

---

## 📌 Overview

**Adversarial AI** studies how machine-learning models can be intentionally manipulated through carefully crafted inputs that cause incorrect predictions — often with *extreme confidence*.

> **The model isn’t confused. It’s confidently wrong.**

Modern ML systems do not understand images, text, or language the way humans do.  
They operate in **high-dimensional mathematical spaces**, optimizing statistical patterns rather than semantic meaning.

That gap is exploitable.

This repository focuses on **black-box adversarial attacks** and why they represent both a **security risk** and an **intellectual-property exposure** in real-world deployments.

---

## 🎯 Threat Model Assumptions

Unless explicitly stated otherwise, all attacks in this repository assume:

- ❌ No access to model weights or gradients  
- ❌ No access to training data  
- ✅ Interaction only through a deployed inference API  

Model output is either:
- confidence / logits (**score-based**), or  
- top-1 label only (**decision-based**)  

These assumptions reflect **production systems**, not academic white-box settings.

---

## 🧪 A Minimal Adversarial Example (For Intuition)

> ⚠️ **Conceptual example**  
> This illustrates *why* adversarial inputs exist.  
> It is **not** the primary threat model explored here.

```python
import torch
import torch.nn.functional as F

model.eval()

image = image.clone().detach().requires_grad_(True)

logits = model(image)
loss = F.cross_entropy(logits, target_label)
loss.backward()

epsilon = 0.002
adversarial_image = image + epsilon * image.grad.sign()
adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)
```

Inference:

```python
prediction = model(adversarial_image).softmax(dim=1)
print(prediction.argmax(), prediction.max())
```

Same image to a human.  
Different reality to the model.

![Adversarial example](images/adversarial_example.png)

---

## 🧭 Why This Works: Decision Boundaries

Neural networks make decisions using **decision boundaries** — invisible mathematical surfaces separating classes in high-dimensional space.

Crossing the boundary does **not** require changing what the input *is* — only where it lands mathematically.

```python
# Conceptual illustration (not runnable logic)
if f(x) > decision_boundary:
    return "dog"
else:
    return "not dog"
```

Small, targeted perturbations push inputs just across these boundaries, producing large classification errors.

![Decision boundary visualization](images/decision_boundary.png)

---

## 🧩 It’s Not Just Images

### 📝 Text Systems (LLMs, moderation)

```text
"I hate you"   → flagged
"I h@te you"   → passes
```

---

### 💳 Rule-Based Fraud Thresholds

```python
amount = 999.99   # allowed
amount = 1000.01  # flagged
```

---

### 🧪 Training-Time Data Poisoning

```python
# Conceptual poisoning example
x_train = np.concatenate([x_clean, x_poisoned])
y_train = np.concatenate([y_clean, y_targeted])
```

Once learned, poisoned behavior is persistent.

---

## ⚠️ Why Black-Box Attacks Matter More Than White-Box

White-box attacks require access to gradients or weights and usually imply:

- insider threats  
- compromised infrastructure  
- catastrophic security failure  

Black-box attacks require **none** of that.

They work against **properly deployed systems** using only standard inference access.

![Black-box attack diagram](images/black_box_attack.png)

---

## 📉 Score-Based Black-Box Attacks (SimBA-Style)

**Threat model:** the attacker observes confidence scores or logits.

The model is treated as a **scoring oracle**.  
The attacker iteratively searches for perturbations that reduce the score of the true class.

### Simplified SimBA-Style Implementation

```python
import torch
import numpy as np

model.eval()

eta = 0.005
num_directions = 1024

# Random perturbation basis
directions = torch.randn(
    (num_directions,) + img.shape,
    device=img.device
)
directions = directions / (
    directions.flatten(1).norm(dim=1).view(num_directions, 1, 1, 1)
)
directions *= eta

perturbation = torch.zeros_like(img)

with torch.no_grad():
    logits = model(img)
    true_label = logits.argmax(1).item()
    best_score = logits[0, true_label].item()

current_label = true_label

while current_label == true_label:
    idx = np.random.randint(num_directions)
    candidate = perturbation + directions[idx]

    with torch.no_grad():
        out = model(img + candidate)
        score = out[0, true_label].item()
        current_label = out.argmax(1).item()

    if score < best_score:
        perturbation = candidate
        best_score = score
```

🔑 **Key insight:**  
Confidence values alone leak enough signal to approximate decision boundaries.

---

## 🧮 Decision-Based Black-Box Attacks (HopSkipJump-Style)

**Threat model:** the attacker sees *only* the predicted label.

The model is treated as a **binary oracle**.

---

### Step 1 — Find Any Misclassification

```python
def find_initial_adversarial(x, y, model, max_trials=2000):
    for _ in range(max_trials):
        noise = 0.2 * torch.randn_like(x)
        x_try = torch.clamp(x + noise, 0.0, 1.0)
        if model(x_try).argmax(1).item() != y:
            return x_try
    raise RuntimeError("Failed to find initial adversarial example")
```

---

### Step 2 — Binary Search to the Boundary

```python
def binary_search_boundary(x, x_adv, y, model, steps=20):
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        x_mid = (1 - mid) * x + mid * x_adv
        if model(x_mid).argmax(1).item() != y:
            hi = mid
        else:
            lo = mid
    return (1 - hi) * x + hi * x_adv
```

---

### Step 3 — Estimate Direction Using Labels Only

```python
def estimate_direction(x_boundary, y, model, eps=0.01, samples=64):
    noise = torch.randn(
        (samples,) + x_boundary.shape[1:],
        device=x_boundary.device
    )
    noise = noise / (
        noise.flatten(1).norm(dim=1).view(samples, 1, 1, 1)
    )

    probes = torch.clamp(x_boundary + eps * noise, 0.0, 1.0)

    with torch.no_grad():
        preds = model(probes).argmax(1)

    signs = torch.where(preds != y, 1.0, -1.0).view(samples, 1, 1, 1)
    grad_est = (signs * noise).mean(dim=0)

    return grad_est / grad_est.norm()
```

By observing **where labels flip**, attackers recover gradient-like information without gradients.

---

## 🔐 Why This Is Also an IP Problem

Black-box attacks enable:

- decision-boundary reconstruction  
- surrogate model training  
- inference of training-data properties  
- replication of proprietary behavior  

No breach required.  
No malware required.  
Just scale.

Inference endpoints become **leakage channels**.

---

## 🛡️ Defense Philosophy

Defense is not about stopping attacks entirely.

It is about **controlling information flow and attacker economics**.

### Practical Controls

- Remove confidence / logit outputs  
- Enforce long-horizon query budgets  
- Detect boundary-seeking behavior  
- Introduce stochastic preprocessing  
- Use adversarial training strategically  
- Tier model access by trust level  
- Treat inference APIs as IP-sensitive assets  
- Explicitly document residual risk  

---

## 🧠 Core Takeaway

> **Decision boundaries are discoverable through normal use.**

Adversarial robustness is not just an ML problem.  
It is a **security, platform, and legal problem**.

---

## 📚 Sources & Influences

- *Adversarial AI Attacks, Mitigations, and Defense Strategies* — John Sotiropoulos (Packt, ISBN 978-1835087985)  
- NVIDIA Deep Learning Institute — *Exploring Adversarial Machine Learning*

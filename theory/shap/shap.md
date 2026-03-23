**SHAP (SHapley Additive exPlanations)**

* Explainable AI method based on **game theory (Shapley values)**
* Explains **how each feature contributes to a prediction**
* Treats features like “players” contributing to the final output

---

**Core idea**
Prediction = base value + sum of feature contributions

---

**What SHAP tells you**

* Which features **increase prediction**
* Which features **decrease prediction**
* **How much each feature contributes**

---

**Key properties**

* **Additive** → contributions sum to output
* **Consistent** → fair importance distribution
* **Model-agnostic** (works with any model)

---

**Types of SHAP explainers**

* TreeExplainer → trees (fast, best)
* KernelExplainer → any model (slow)
* DeepExplainer → neural networks

---

**SHAP Summary Plot**

* Visualizes **feature impact across entire dataset**
* Shows both **importance + direction of effect**

---

**How to read summary plot**

* Each row = feature

* Each dot = one data point

* **X-axis (SHAP value)**

  * Left (−) → decreases prediction
  * Right (+) → increases prediction

* **Color**

  * Red → high feature value
  * Blue → low feature value

* **Order (top → bottom)**

  * Most important → least important

---

**Interpretation rules**

* Red on right → high value increases output
* Red on left → high value decreases output
* Blue on right → low value increases output
* Blue on left → low value decreases output

---

**Importance indicator**

* Wider spread → more impact
* Narrow/near 0 → less important

---

**Types of summary plots**

* Beeswarm → detailed (default)
* Bar plot → only importance

---

**Example**
Model predicts price:

* Size → +20 (increases price)
* Age → −5 (decreases price)

---

**Use cases**

* Model debugging
* Feature importance
* Explain predictions
* Detect bias

---

**One-line definitions**

* **SHAP**: method to explain feature contribution using Shapley values
* **Summary plot**: visualization showing impact of all features on predictions across dataset

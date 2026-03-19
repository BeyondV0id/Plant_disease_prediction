# Voting and Stacking Classifier

## 1. Voting Classifier

A `VotingClassifier` combines predictions from multiple models and gives one final prediction.

### Hard voting
- Each model predicts a class.
- The class with the most votes wins.

Example:
- Model A -> `early_blight`
- Model B -> `healthy`
- Model C -> `early_blight`

Final prediction: `early_blight`

### Soft voting
- Each model gives class probabilities.
- The probabilities are averaged.
- The class with the highest average probability wins.

Example for `early_blight`:
- Model A -> `0.80`
- Model B -> `0.60`
- Model C -> `0.90`

Average:

```text
(0.80 + 0.60 + 0.90) / 3 = 0.7667
```

If this is the highest among all classes, final prediction is `early_blight`.

## 2. Stacking Classifier

A `StackingClassifier` also combines multiple models, but not by simple voting.

It works in 2 levels:

1. Base models make predictions.
2. A meta-model learns from those predictions.

So:

```text
original data -> base models -> meta-model -> final prediction
```

## 3. If Logistic Regression is the Meta-Model

Suppose you have 3 base models.

For one image, they predict probability for `early_blight` as:

- Model 1 -> `0.80`
- Model 2 -> `0.60`
- Model 3 -> `0.90`

Then the meta input becomes:

```python
X_meta = [0.80, 0.60, 0.90]
```

If the true class of that image is `early_blight`, then:

```python
y_meta = "early_blight"
```

If label encoding is used, maybe:

```python
healthy = 0
early_blight = 1
late_blight = 2
```

Then:

```python
y_meta = 1
```

## 4. Important Point About `y`

`y` is not a probability.

`y` is the true label.

So for an `early_blight` sample:

```python
X_meta = [0.80, 0.60, 0.90]
y_meta = "early_blight"   # or 1 after encoding
```

## 5. Do We Need To Convert `X_meta` Into One Value?

No.

`X_meta` can stay as multiple values.

Logistic Regression can directly train on multiple input features.

So it learns something like:

```text
score = w1*(0.80) + w2*(0.60) + w3*(0.90) + b
```

That means:
- `w1` = weight for model 1
- `w2` = weight for model 2
- `w3` = weight for model 3

The meta-model learns which base model should be trusted more.

## 6. Training View

The meta-model training table may look like this:

| `X_meta` from base models | `y_meta` true label |
|---|---|
| `[0.80, 0.60, 0.90]` | `early_blight` |
| `[0.10, 0.20, 0.05]` | `healthy` |
| `[0.30, 0.25, 0.85]` | `late_blight` |

So Logistic Regression is trained on:
- Input: predictions from base models
- Output: actual disease label

## 7. Easy Way To Remember

### Voting
Many models vote, final answer is chosen directly.

### Stacking
Many models predict first, then another model learns how to combine them.

### In your case
- Base models give prediction values
- Those values form `X_meta`
- True class like `early_blight` is `y_meta`
- Logistic Regression learns the final decision

## 8. One-Line Summary

In stacking, Logistic Regression is trained on the predictions of base models, and the target `y` is still the actual class label such as `early_blight`.

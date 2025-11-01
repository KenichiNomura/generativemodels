# Integer Discrete Flow (IDF) Class: Detailed Explanation

## Purpose
The `IDF` class implements Integer Discrete Flows for modeling discrete data (e.g., images with integer pixel values). It provides invertible transformations for likelihood-based generative modeling and sampling.

---

## Key Attributes
- **Translation Nets:**
  - `self.t` (for idf_git=1): List of neural nets, each mapping from $(B, D/2)$ to $(B, D/2)$.
  - `self.t_a`, `self.t_b`, `self.t_c`, `self.t_d` (for idf_git=4): Each is a list of nets mapping from $(B, 3D/4)$ to $(B, D/4)$.
- **Prior Parameters:**
  - `self.mean`: $(1, D)$ learnable mean vector.
  - `self.logscale`: $(1, D)$ learnable log-scale vector.
- **Other:**
  - `self.num_flows`: Number of flow steps (layers).
  - `self.D`: Input dimension (e.g., $D=64$ for $8\times8$ images).
  - `self.round`: Straight-through rounding function.

---

## Constructor (`__init__`)
- **Input:**
  - `netts`: List of net constructors (length 1 or 4).
  - `num_flows`: Number of flow steps.
  - `D`: Input dimension.
- **Behavior:**
  - If `len(netts)==1`, sets up classic coupling (splits input into two halves).
  - If `len(netts)==4`, sets up 4-partition coupling (splits input into four quarters).
  - Initializes prior parameters and rounding function.

---

## Methods & Tensor Shapes
### 1. `coupling(x, index, forward=True)`
- **idf_git=1:**
  - Input: $x$ of shape $(B, D)$
  - Split: $(xa, xb)$ each $(B, D/2)$
  - Net: $t[index](xa)$ produces $(B, D/2)$
  - Output: Concatenate $(xa, yb)$ → $(B, D)$
- **idf_git=4:**
  - Input: $x$ of shape $(B, D)$
  - Split: $(xa, xb, xc, xd)$ each $(B, D/4)$
  - Each net takes $(B, 3D/4)$ and outputs $(B, D/4)$
  - Output: Concatenate $(ya, yb, yc, yd)$ → $(B, D)$

### 2. `permute(x)`
- Flips the feature dimension: $x.flip(1)$
- Input/Output: $(B, D)$

### 3. `f(x)`
- Applies `coupling` and `permute` for `num_flows` steps.
- Input/Output: $(B, D)$

### 4. `f_inv(z)`
- Applies inverse permutation and coupling in reverse order.
- Input/Output: $(B, D)$

### 5. `forward(x, reduction='avg'|'sum')`
- Computes $z = f(x)$, then negative log-likelihood under prior.
- Input: $(B, D)$
- Output: Scalar (mean or sum over batch)

### 6. `sample(batchSize)`
- Samples $z$ from prior: $(B, D)$
- Applies $f^{-1}(z)$: $(B, D)$
- Returns: $x.view(B, 1, D)$ (for plotting as images)

### 7. `log_prior(x)`
- Computes log-probability for each sample.
- Input: $(B, D)$
- Output: $(B,)$ (sum over features)

### 8. `prior_sample(batchSize, D)`
- Samples from logistic prior, rounds to integer.
- Output: $(B, D)$

---

## Example Shapes (B=64, D=64)
- **idf_git=1:**
  - $x$: $(64, 64)$
  - $xa, xb$: $(64, 32)$
  - $t[index](xa)$: $(64, 32)$
  - Output: $(64, 64)$
- **idf_git=4:**
  - $x$: $(64, 64)$
  - $xa, xb, xc, xd$: $(64, 16)$
  - Each net input: $(64, 48)$
  - Each net output: $(64, 16)$
  - Output: $(64, 64)$

---

## Special Notes
- **Straight-Through Rounding:**
  - Rounding is used in forward pass, but gradients flow through unchanged in backward pass.
- **D Requirements:**
  - For idf_git=4, $D$ must be divisible by 4.
- **Broadcasting:**
  - `mean` and `logscale` are broadcasted to match input shape.
- **Sampling/Plotting:**
  - Output of `sample` is reshaped for image plotting.

---

Let me know if you want a step-by-step walkthrough of a specific method or more details on edge cases!

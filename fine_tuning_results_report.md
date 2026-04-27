# Fine-Tuning Experiment Results Report

## Overview
This report summarizes the results of three different fine-tuning experiments: **Baseline Validation**, **Evolutionary Strategy (ES) Trial 011**, and **Particle Swarm Optimization (PSO) Trial 011**. The objective was to evaluate different hyperparameter configurations and their impact on model performance and training efficiency.

All models were evaluated on a validation set of 128 examples, targeting the `q_proj` and `v_proj` modules with a cosine learning rate scheduler.

---

## 📊 Summary Comparison

| Metric | Baseline Validation | ES Trial 011 | PSO Trial 011 |
| :--- | :--- | :--- | :--- |
| **Validation Score** | 0.3281 | **0.3984** 🏆 | 0.3594 |
| **Training Loss** | 0.2369 | **0.0756** | 0.2926 |
| **Runtime (Seconds)** | 3249.31 | 5487.24 | **1491.72** ⚡ |
| **LoRA Rank (r)** | 4 | 16 | 8 |
| **LoRA Alpha** | 16 | 64 | 64 |
| **Learning Rate** | 0.0002 | 0.0005 | 0.0002 |
| **Batch Size** | 8 | 8 | 4 |
| **Steps** | 300 | 400 | 200 |
| **Dropout** | 0.05 | 0.10 | 0.15 |

---

## 🔍 Detailed Experiment Results

### 1. Baseline Validation
* **Path:** `outputs/baseline_validation/result.json`
* **Performance:** Achieved the lowest validation score among the three runs, serving as the benchmark. 
* **Key Configuration:** Used a low LoRA rank (4) and alpha (16) with standard dropout (0.05) over 300 steps.

### 2. ES Trial 011 (Evolutionary Strategy)
* **Path:** `outputs/es/trial_011/result.json`
* **Performance:** **Highest validation score (0.3984)** and best convergence (lowest training loss of 0.0756). However, it required the longest training time (~1.5 hours).
* **Key Configuration:** Increased capacity with LoRA rank 16 and alpha 64. Higher learning rate (0.0005) and longer training duration (400 steps).

### 3. PSO Trial 011 (Particle Swarm Optimization)
* **Path:** `outputs/pso/trial_011/result.json`
* **Performance:** A balanced run that achieved a better validation score than the baseline (0.3594) while being significantly faster (**~24.8 minutes**). Training loss remained relatively high, indicating it might benefit from more steps.
* **Key Configuration:** Moderate capacity (Rank 8, Alpha 64), smaller batch size (4), and fewer steps (200). Highest dropout rate (0.15).

---

## 💡 Key Takeaways
1. **Best Performance:** The **ES Trial 011** configuration yielded the best model accuracy (~39.8%), demonstrating that a higher LoRA rank/alpha combined with more steps significantly improves outcomes, albeit at the cost of longer runtime.
2. **Best Efficiency:** The **PSO Trial 011** run was highly efficient, running over 3.6x faster than the ES trial and 2.1x faster than the baseline, while still outperforming the baseline validation score.
3. **Capacity Matters:** Increasing the LoRA rank from 4 (baseline) to 8 (PSO) and 16 (ES) consistently correlated with improved validation scores.

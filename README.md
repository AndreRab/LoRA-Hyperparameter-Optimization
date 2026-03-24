# Hyperparameter Optimization for PEFT (LoRA / Adapters) on a Small Instruction LLM

## Project goal

This project studies **hyperparameter optimization (HPO)** for **parameter-efficient fine-tuning (PEFT)** of a small **instruction-tuned LLM** using **LoRA**. The main objective is to compare budget-aware optimization methods such as **Evolution Strategies (ES)** and **Particle Swarm Optimization (PSO)** under a limited search budget (for example **20–40 evaluated configurations**).

The final outcome should answer:

- Which HPO method finds better LoRA hyperparameters under the same budget?
- How much improvement do we get over a strong baseline configuration?
- Which hyperparameters matter most for short PEFT runs?

---

## Datasets

### 1. Super-NaturalInstructions (SuperNI)

SuperNI is a large benchmark of **1,616 NLP tasks** covering many task types and languages, designed for instruction-following and cross-task generalization. It is suitable when the project goal is to optimize LoRA for **general instruction tuning** rather than for one narrow task.

- Dataset / repo: <https://github.com/allenai/natural-instructions>
- Paper: <https://arxiv.org/abs/2204.07705>

### 2. GSM8K

GSM8K is a benchmark of about **8.5K grade-school math word problems**. It is suitable when the project goal is to optimize LoRA for **mathematical reasoning**. The original paper reports roughly **7.5K training** problems and **1K test** problems.

- Dataset: <https://huggingface.co/datasets/openai/gsm8k>
- Paper: <https://arxiv.org/abs/2110.14168>

### Mark
To avoid overfitting only on one dataset, will be used a mix of them.

---

## Models

Use an **instruction-tuned** model in the **3B** or **7B** class.

### Good Hugging Face candidates

#### ~3B
- **Qwen/Qwen2.5-3B-Instruct**
- **microsoft/Phi-3-mini-4k-instruct** *(3.8B, close to 3B)*

#### ~7B
- **mistralai/Mistral-7B-Instruct-v0.3**
- **meta-llama/Llama-3.1-8B-Instruct** *(8B, close to 7B)*

Recommended strict-size choices:

- **3B:** `Qwen/Qwen2.5-3B-Instruct`
- **7B:** `mistralai/Mistral-7B-Instruct-v0.3`

---

## PEFT method

### LoRA

LoRA freezes the original model weights and injects trainable low-rank matrices into selected Transformer layers, greatly reducing the number of trainable parameters and memory usage during fine-tuning.

- LoRA paper: <https://arxiv.org/abs/2106.09685>

---

## Hyperparameters to optimize

The genotype (search space) may include:

- **LoRA rank** `r`
- **LoRA alpha**
- **LoRA dropout**
- **learning rate**
- **batch size**
- **number of training steps**
- **scheduler type**
- optionally: **target layers / modules** where LoRA is attached

### Example search space

```text
rank:            [4, 8, 16, 32, 64]
alpha:           [8, 16, 32, 64, 128]
dropout:         [0.0, 0.05, 0.1, 0.15]
learning rate:   [5e-6 ... 5e-4] (log-scale)
batch size:      [2, 4, 8, 16]
steps:           [100, 200, 300, 500]
scheduler:       [linear, cosine, constant]
target modules:  predefined module sets
```

---

## Optimization methods

### 1. Evolution Strategies (ES)

ES works as follows:

1. Generate several candidate hyperparameter sets.
2. Evaluate them.
3. Keep the better ones.
4. Create new variants by slightly mutating them.
5. Repeat.

Short intuition:

> **selection + mutation**

Good configurations survive, and the next candidates are created as slightly modified versions of them.

### 2. Particle Swarm Optimization (PSO)

PSO works as follows:

1. Start with many candidate configurations.
2. Each one remembers its own best result.
3. All of them also know the global best result.
4. Each candidate moves toward:
   - its own best point,
   - the swarm’s best point.
5. Repeat.

Short intuition:

> **move candidate solutions toward promising regions**

---

## Budget-aware setup

This is a **budgeted HPO** experiment.

Suggested limits:

- **20–40 total evaluated configurations**
- **short fine-tuning runs only**
- each run should be capped by **time** or **step budget**

Example:

- **20 particles / individuals total**, or
- **5–10 per generation / iteration**, until the global budget is exhausted

A budget-aware setup is important because PEFT on 3B–7B models is still expensive even with LoRA.

---

## Data split

Use three disjoint splits:

- **train** — used for short PEFT training during HPO
- **validation** — used to compare hyperparameter configurations
- **test** — used only once for the final comparison after HPO

### Remarls

1. Create `train`, `validation`, and `test` splits.
2. Run ES and PSO on **train -> validation** only.
3. Select the best hyperparameters according to the **validation metric**.
4. Retrain the model using the chosen hyperparameters.
5. Compare the final tuned model against a baseline on the **test set**.

---

## Validation metric

Use a **task metric**, not only training or validation loss.

### Metric for GSM8K

Use:

- **Exact Match (EM)** on the **final numeric answer**

Reason:

- GSM8K is a reasoning benchmark with a clear final answer.
- EM is easy to interpret.
- It reflects actual task performance better than loss.

Optional secondary metrics:

- answer extraction success rate
- token-level accuracy for the final answer span

### Metric for SuperNI

Because SuperNI contains many task types, a single universal metric is not ideal. A practical choice is:

- **task-level score averaged across tasks**

Depending on task type, this can include:

- **Exact Match / Accuracy** for classification-style tasks
- **ROUGE-L** for generative tasks
- **Macro-average across normalized task scores**

---

## Entire pipeline

### Step 1. Mix the dataset

### Step 2. Prepare the splits

Split data into:

- `train`
- `validation`
- `test`

If needed, create smaller subsets for quick experiments.

### Step 3. Choose the base model

Pick one instruction model, for example:

- `Qwen/Qwen2.5-3B-Instruct`, or
- `mistralai/Mistral-7B-Instruct-v0.3`

### Step 4. Define the search space

Define the possible values of:

- rank
- alpha
- dropout
- learning rate
- batch size
- steps
- scheduler
- target modules

### Step 5. Define the baseline

Prepare one hand-designed baseline configuration.

Example baseline:

```text
rank = 16
alpha = 32
dropout = 0.05
learning rate = 2e-4
batch size = 8
steps = 300
scheduler = cosine
```

### Step 6. Run HPO

For each optimization method (**ES** and **PSO**):

1. Generate candidate configurations.
2. Train each candidate for a **short run**.
3. Evaluate on the **validation set** using the selected **task metric**.
4. Update the optimizer state.
5. Repeat until the evaluation budget is exhausted.

### Step 7. Select the best hyperparameters

Choose the configuration with the best **validation score**.

### Step 8. Final training

Retrain using the selected hyperparameters.

Two practical variants are possible:

- retrain on **train only** and report on **test**
- retrain on **train + validation** and report once on **test**

### Step 9. Final comparison

Compare:

- **baseline LoRA configuration**
- **best ES configuration**
- **best PSO configuration**

Report:

- validation best score
- final test score
- total number of evaluated configurations
- approximate compute cost / runtime

---

## Suggested final experimental design

A clean version of the project could be:

- **Dataset:** GSM8K
- **Model:** Qwen2.5-3B-Instruct
- **PEFT:** LoRA
- **Metric:** Exact Match on final answer
- **HPO methods:** ES and PSO
- **Budget:** 30 total configurations per method
- **Training per configuration:** 200–500 steps or max 10 minutes

This design is simple, reproducible, and easy to explain.

---

## Useful papers to read before implementation

### Core paper
- **LoRA: Low-Rank Adaptation of Large Language Models** — <https://arxiv.org/abs/2106.09685>

### Dataset papers
- **Super-NaturalInstructions** — <https://arxiv.org/abs/2204.07705>
- **GSM8K / Training Verifiers to Solve Math Word Problems** — <https://arxiv.org/abs/2110.14168>

### Useful background on PEFT
- **Parameter-Efficient Fine-Tuning for Large Models** — survey paper: <https://arxiv.org/pdf/2403.14608>1
- **Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment** — <https://arxiv.org/html/2312.12148v1>

### HPO paper mentioned in the task
- **Hyperparameter Optimization for Large Language Model Instruction-Tuning** — <https://arxiv.org/html/2312.00949v2>

---

## Minimal checklist

- [ ] Choose one dataset: SuperNI or GSM8K
- [ ] Choose one base model: 3B or 7B
- [ ] Prepare train / validation / test splits
- [ ] Define the LoRA search space
- [ ] Define the baseline configuration
- [ ] Implement ES
- [ ] Implement PSO
- [ ] Use a task metric on validation, not only loss
- [ ] Run the HPO budget
- [ ] Retrain best configuration
- [ ] Compare against baseline on test

---

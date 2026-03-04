# On-Policy Self-Distillation Implementation

## Overview

This repository contains an experimental implementation comparing a baseline training approach with an On-Policy Self-Distillation (OPSD) based training method for improving reasoning capabilities in large language models.

The project explores whether a language model can improve its reasoning ability by learning from its own generated outputs while leveraging ground-truth solutions as privileged information.

Two different training approaches are provided for comparison:

* `baseline_model.ipynb` – Standard baseline training implementation
* `opsd_kl_distillation.ipynb` – Implementation of On-Policy Self-Distillation using token-level KL divergence

The goal of this repository is to demonstrate the differences between traditional supervised training and self-distillation based learning.

---

## Repository Structure

```
.
├── baseline_model.ipynb
├── opsd_kl_distillation.ipynb
├── requirements.txt
└── README.md
```

**baseline_model.ipynb**
Implements a standard training pipeline used as the experimental baseline.

**opsd_kl_distillation.ipynb**
Implements the On-Policy Self-Distillation training pipeline where a model learns from its own generated outputs.

**requirements.txt**
Contains the Python dependencies required to run the notebooks.

---

## Methodology

### Baseline Model

The baseline notebook follows a conventional training setup where the model is trained directly on problem–solution pairs using supervised learning.

Characteristics of this approach:

* Uses fixed dataset trajectories
* Trains directly on ground truth reasoning steps
* Does not perform on-policy generation
* Serves as the reference model for comparison

---

### On-Policy Self-Distillation (OPSD)

In the OPSD method, the same language model is used as both a **teacher** and a **student**, but under different input contexts.

The student policy receives only the problem prompt and generates a solution.
The teacher policy receives the problem along with the reference solution.

During training:

1. The student generates a response
2. Both teacher and student evaluate the generated sequence
3. A token-level divergence loss is computed between their probability distributions
4. Gradients update the student policy

This allows the model to improve its reasoning by aligning its outputs with a privileged teacher distribution.

---

Each notebook contains the full experimental workflow including:

* dataset loading
* model initialization
* training loop
* evaluation pipeline

---

## Experimental Goal

The main objective of this repository is to compare:

* Traditional supervised fine-tuning
* On-Policy Self-Distillation training

The experiments aim to observe whether self-distillation can improve reasoning performance while maintaining training efficiency.

---

## Notes

* Training large language models may require GPUs with sufficient memory.
* The notebooks are structured to clearly show the difference between baseline training and the OPSD approach.

# Diabetic Retinopathy Screening Demo

**Team 6thSense** — Elijah, KaiXi, Garet, Isaac  
National AI Competition 2026 — Computing Track

A deep learning demo that classifies retinal fundus images into 5 DR severity grades using a ConvNeXt-Tiny model trained with 5-fold cross-validation.

## Features

- Upload any retinal fundus image for instant DR grading
- Confidence scores for all 5 severity classes
- Grad-CAM heatmap showing which regions the model focused on
- Test-Time Augmentation (4 geometric flips) for robust predictions

## Model Performance

| Metric | Score |
|--------|-------|
| OOF Macro F1 | 0.7636 |
| OOF Accuracy | 85.69% |
| Architecture | ConvNeXt-Tiny (28.6M params) |
| Input Resolution | 512×512 |

## Disclaimer

This is a research prototype built for the National AI Competition 2026. It is **NOT** a medical diagnosis tool and should not be used for clinical decision-making.

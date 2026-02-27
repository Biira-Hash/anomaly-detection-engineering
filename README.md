# Anomaly Detection Engineering Pipeline

## Overview

This project implements a structured and modular machine learning pipeline for anomaly detection. The objective was to transform a notebook-based experimental workflow into a production-style engineering pipeline with proper architecture and reproducibility.

This repository demonstrates core software engineering and ML engineering principles:

- Modular architecture
- Separation of concerns
- Reproducible environment setup
- Command-line training pipeline
- Clean version control practices

---

## Project Structure

anomaly-detection-engineering/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│ └── experimentation.ipynb
│
└── src/
├── data_loader.py
├── preprocessing.py
├── model.py
└── train.py


---

## Architecture Overview

The project is organized into independent, reusable modules:

### data_loader.py
Responsible for loading datasets from disk. Isolates I/O logic from training logic.

### preprocessing.py
Handles feature-target separation and missing value handling.

### model.py
Defines model creation, training, and evaluation logic.

### train.py
Acts as the pipeline orchestrator, connecting all components and providing a CLI interface.

### notebooks/experimentation.ipynb
Contains exploratory data analysis and initial experimentation.

---

## Training Pipeline Flow

The pipeline executes the following steps:

1. Load dataset  
2. Split features and target  
3. Handle missing values  
4. Perform train-validation split  
5. Create model  
6. Train model  
7. Evaluate model using F1 score  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Biira-Hash/anomaly-detection-engineering.git
cd anomaly-detection-engineering
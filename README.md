# Model Optimization Pipeline

A complete deep learning compression pipeline implementing multiple model optimization strategies including:

- Structured Pruning
- Unstructured Pruning
- Quantization-Aware Training (QAT)
- Full INT8 TFLite Quantization
- Model size benchmarking (raw + gzipped)
- Accuracy comparison across strategies

---

## 🚀 Project Overview

This repository demonstrates practical model compression techniques using:

- TensorFlow
- TensorFlow Model Optimization Toolkit (TF-MOT)
- TFLite INT8 conversion

The goal is to compare:

- Accuracy retention
- Model size reduction
- Parameter count
- Compression efficiency

---

## 🧠 Implemented Strategies

### 1️⃣ Baseline Model
Standard CNN model trained normally.

### 2️⃣ Structured Pruning
Removes entire filters or channels.

### 3️⃣ Unstructured Pruning
Removes individual weights based on sparsity target.

### 4️⃣ Quantization-Aware Training (QAT)
Simulates quantization during training.

### 5️⃣ Post-Training Quantization
Converts trained model to TFLite INT8.

---

## 📊 Metrics Reported

For each strategy:

- Accuracy
- Raw model size (KB)
- Gzipped model size (KB)
- Parameter count
- Convolution filter count

---

## 📂 Project Structure
```bash
Model_Optimization
├── .gitignore
├── environment.yml                  # Environment file
├── notebook
│   └── ModelOptimization.ipynb      # Notebook version of the pipeline
├── README.md
├── requirements.txt                 # Packages required
├── scripts
│   └── run_pipeline.py              # To run the whole pipeline
└── src
    ├── __init__.py
    ├── configs
    │   └── compression.yaml         # Model configuration file
    ├── data
    │   └── mnist.py                 # Data loader
    ├── evaluation
    │   └── test.py                  # Testing the model
    ├── models
    │   ├── base_model.py            # Base line model architecture
    │   ├── clustering.py            # Clustered model
    │   ├── quantization.py          # Quantized model
    │   ├── structured.py            # Structured pruning architecture
    │   └── unstructured.py          # Unstructured pruning model
    ├── pipeline
    │   └── compression_pipeline.py  # Pipeline code
    ├── training
    │   └── trained.py               # Training code
    └── utils
        ├── comparision.py            
        ├── env.py
        ├── filters.py
        ├── logging.py
        ├── size.py
        └── sparsity.py
```

---

## ⚙️ Installation (Simple & Stable Method)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/kja8586/Model_Optimization.git
cd Model_Optimization
```

### 2️⃣ Create virtual environment
# ⚙️ Environment Setup

We recommend using **Conda** for maximum stability and reproducibility.

---

## 🥇 Option 1 — Conda (Recommended)

This ensures:

- Correct Python version
- Exact dependency versions
- Clean isolated environment
- Minimal dependency conflicts

---

### 1️⃣ Create environment from `environment.yml`

```bash
conda env create -f environment.yml
```

### 2️⃣ Activate environment

```bash
conda activate mo
```

### 3️⃣ Run the pipeline

```bash
python -m scripts.run_pipeline
```

---

#### 📄 Example `environment.yml`

```yaml
name: mo
channels:
  - defaults
dependencies:
  - python=3.10.14
  - pip
  - pip:
      - tensorflow==2.19.0
      - tf_keras==2.19.0
      - tensorflow-model-optimization==0.8.0
      - pandas==2.2.2
      - numpy==1.26.0
      - PyYAML==6.0.2
```

---

#### 🥈 Option 2 — Virtual Environment (Lightweight Alternative)

Use this if you prefer standard Python environments.

---

### 1️⃣ Create virtual environment

```bash
python3.10 -m venv venv
```

### 2️⃣ Activate it

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run

```bash
python -m scripts.run_pipeline
```

---

# 🎯 Recommendation

For this project (TensorFlow + TF-MOT + TFLite):

👉 **Conda is strongly recommended** for better reproducibility and fewer dependency conflicts.


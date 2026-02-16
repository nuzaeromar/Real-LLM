<div align="center">

# Leveraging Pre-Trained Language Models for Realistic Adversarial Attacks \

---

<p>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
  </a>
</p>

### **Nuzaer Omar**<sup>1</sup>, Ademola Adesokan<sup>2</sup>, Sanjay Madria<sup>1</sup>  

<sup>1</sup>Missouri University of Science & Technology  
<br>
<sup>2</sup>University of Central Arkansas  

---

</div>

Official Repository for "Leveraging Pre-Trained Language Models for Realistic Adversarial Attacks" published in "2025 IEEE International Conference on Big Data". 

---

## 📌 What This Repository Covers

- Adversarial attack pipelines for multiple transformer models
- Evaluation of robustness across different pretrained checkpoints
- Modular attack implementations for easy comparison
- Reproducible experiments for NLP security research

---

## 🧠 Supported Models & Attacks

The repository includes attack implementations for the following models:

- **BERT**
- **DistilBERT**
- **RoBERTa**
- **DistilRoBERTa**
- **ALBERT**
- **Large BERT variants**  

Each model has its own implementation containing:
- Attack scripts
- Configuration details
- Evaluation utilities

- The init.py either needs to be replaced inside the attacker folder of OpenAttack or
- A new attacker needs to included in the OpenAttack framework.
---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/nuzaeromar/Real-LLM.git
cd Real-LLM
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run an attack experiment  
Example (model-specific scripts may vary):

```bash
python realllm_test.py
```

## 👤 Author
Nuzaer Omar  
PhD Candidate, Computer Science  
Missouri University of Science & Technology



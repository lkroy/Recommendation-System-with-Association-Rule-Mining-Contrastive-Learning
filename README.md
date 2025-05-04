# Recommendation-System-with-Association-Rule-Mining-Contrastive-Learning



PrecisionRec is a hybrid recommendation system developed at **National Institute of Technology, Patna** to enhance personalization in retail and e-commerce. It integrates **association rule mining** (positive and negative rules) with **contrastive learning** to generate high-quality item embeddings, used in a **Transformer-based SASRec** model for sequential recommendations.

---

## 🛠️ Tech Stack

- Python 3.8+
- PyTorch
- Pandas, NumPy, Scikit-learn
- Contrastive Learning
- Transformers (SASRec)

---

## ✨ Features

- **Hybrid Approach**: Combines association rule mining and contrastive learning for robust item embeddings.
- **Sequential Recommendations**: Utilizes SASRec Transformer for accurate next-item prediction.
- **Scalable Design**: Embeddings are modular and compatible with other ML/DL architectures (e.g., GNNs, RNNs).
- **Optimized Training**: Early stopping, learning rate scheduling, and hyperparameter tuning.
- **Evaluation Metrics**: Uses Recall@10 and NDCG@10 for performance measurement.

---

## 📂 Datasets

- **Dunnhumby**: Retail transaction dataset for market basket analysis.
- **Ta-Feng**: Grocery shopping dataset for sequential recommendations.

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/lkroy/PrecisionRec.git
cd PrecisionRec

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Download the datasets (e.g., Dunnhumby, Ta-Feng) and place them in the `data/` directory.

---

## 🚀 Usage

### 1. Preprocess the Dataset

```bash
python preprocess.py --dataset dunnhumby --output data/processed/
```

### 2. Train the Model

```bash
python train.py --model precisionrec --dataset dunnhumby --epochs 50 --batch_size 128
```

### 3. Evaluate the Model

```bash
python evaluate.py --model precisionrec --dataset dunnhumby --metrics recall@10 ndcg@10
```

---

## 🧠 Model Architecture

- **Association Rule Mining**: Extracts positive and negative item relationships using Apriori-based algorithms.
- **Contrastive Learning**: Captures co-occurrence and mutual exclusivity patterns for embedding generation.
- **SASRec Transformer**: Learns sequential patterns of user behavior for next-item recommendations.
- **Optimization**: Uses early stopping and dynamic learning rate scheduling.

---

## 📊 Results

| Dataset    | Recall@10 | NDCG@10 |
|------------|-----------|---------|
| Dunnhumby  | 0.1357    | 0.0906  |
| Ta-Feng    | 0.1324    | 0.0887  |

---

## 🔮 Future Work

- Integrate **Graph Neural Networks (GNNs)** for better item relationship modeling.
- Support **real-time recommendation** using streaming data.
- Extend to **multi-modal data** (text, image) for richer embeddings.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Open an issue for bugs or suggestions.
- Submit a pull request with improvements.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---



---

## 📅 Timeline

Project Duration: **April 2025**

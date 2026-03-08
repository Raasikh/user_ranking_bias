# User-Ranking Model with Bias-Corrected Embeddings & Sequential Context Modeling

## Resume Bullet
> **"User-ranking model accuracy increase of 15% through bias-corrected embeddings and sequential context modeling"**

A runnable, end-to-end project that teaches you exactly what every word in that bullet means — with code you can step through, modify, and explain in an interview.

---

## What Does This Actually Mean? (Plain English)

Imagine building the ranking system for Amazon search. A user searches "wireless headphones" and sees 10 products. They click the **2nd result**.

**Question:** Did they click because it was the *best* headphone? Or because it was near the *top of the page*?

**This is position bias** — users naturally click higher-ranked items, which corrupts training data. A naive model learns "items at position 1 are great!" instead of learning actual quality.

### The Three Key Concepts

| Concept | What It Means | Why It Matters |
|---|---|---|
| **User-Ranking Model** | Predicts which item a user will click/buy from a ranked list | Core of search & recommendation systems |
| **Bias-Corrected Embeddings** | Learns position bias separately, then removes it at inference | Rankings reflect true relevance, not display position |
| **Sequential Context Modeling** | Uses recent click history as a sequence to model evolving intent | Captures "browsing → comparing → ready to buy" |

### The Core Math

```
TRAINING:   P(click) = sigmoid(relevance_score + position_bias)
                       ↑ Model learns BOTH terms from click data

INFERENCE:  P(click) = sigmoid(relevance_score + 0)
                       ↑ Position bias zeroed out → pure quality ranking
```

---

## Project Structure

```
user_ranking_bias_correction.py     # Main script (VS Code + Colab)
README_user_ranking.md              # This file
ranking_model_results.png           # Generated after running
```

---

## How to Run

### Option A: Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. File → Upload Notebook → Upload `user_ranking_bias_correction.py`
3. Run All (Ctrl+F9) — takes ~1-2 minutes, no GPU needed

**Note:** The script uses `# %%` cell markers. Colab treats these as interactive cells automatically.

### Option B: VS Code (Interactive Python)

1. Install the **Jupyter** extension in VS Code
2. Open `user_ranking_bias_correction.py`
3. Each `# %%` creates a runnable cell — click "Run Cell" to step through
4. Dependencies: `pip install numpy pandas scikit-learn matplotlib seaborn`

### Option C: Command Line

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python user_ranking_bias_correction.py
```

---

## What the Code Does (Section by Section)

| Section | What Happens | Key Concept |
|---|---|---|
| **1-2** | Generate synthetic e-commerce click data with position bias | Position bias = users click top items regardless of quality |
| **3** | Build sequential interaction histories per user | Sequential context = order of clicks matters |
| **4** | Feature engineering: baseline vs sequential features | Recency-weighted embeddings capture evolving intent |
| **5** | Train/test split and scaling | Standard ML pipeline |
| **6** | Baseline model: GBT with position features (biased) | The "before" — learns biased patterns |
| **7** | Bias-corrected model: position features removed | Forces model to learn actual item quality |
| **8** | Full model: bias-corrected + sequential context features | The "after" — learns quality + evolving user intent |
| **9** | Test evaluation with AUC, accuracy, NDCG | Ranking-specific metrics |
| **10** | Head-to-head comparison → the ~15% improvement | The headline number |
| **11** | Visualizations: metrics bars, position fairness, feature importance | Evidence you can show |
| **12-13** | Interview Q&A and concept map | What to SAY about this |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  BIAS-CORRECTED SEQUENTIAL RANKER                                │
│                                                                   │
│  ┌───────────────┐     ┌─────────────────┐     ┌──────────────┐ │
│  │ Past 10 Items  │────►│ Recency-Weighted│────►│ User Context │ │
│  │ (sequence)     │     │ Averaging / GRU │     │ Vector       │ │
│  └───────────────┘     └─────────────────┘     └──────┬───────┘ │
│                                                        │         │
│  ┌───────────────┐                                     │         │
│  │ Target Item    │────────────────────────────────────►├─► MLP  │
│  │ Embedding      │                                     │  Head  │
│  └───────────────┘                          ┌──────────┤        │
│                                             │          ▼         │
│  ┌───────────────┐     ┌────────────┐      │     Relevance     │
│  │ Position ID    │────►│ Bias Param │──────┘     + Bias        │
│  └───────────────┘     │ (learned)  │            = P(click)     │
│                        │ REMOVED AT │                            │
│                        │ INFERENCE! │                            │
│                        └────────────┘                            │
└──────────────────────────────────────────────────────────────────┘
```


## Key Metrics to Know

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **NDCG@K** | Are the best items ranked in the top K? | Primary ranking quality metric |
| **AUC-ROC** | Discrimination between clicks and non-clicks | General model quality |
| **Accuracy** | Binary prediction correctness | Easy to communicate ("15% more accurate") |
| **Position-stratified Accuracy** | Accuracy by display position | Proves bias correction works (should be uniform) |

---

## Dependencies

- Python 3.8+
- NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **No GPU, no PyTorch required** (runs on any laptop)

---

## From This Demo → Production

| This Demo | Production Version |
|---|---|
| Sklearn GradientBoosting | PyTorch / TensorFlow |
| Recency-weighted average | GRU / LSTM / Transformer encoder |
| 1.5K users, 400 items | Millions of users and items |
| Synthetic data | Real click logs from search/recommendations |
| Single machine | Distributed training (DeepSpeed/FSDP) |
| Same math, same concepts | Same interview answers |

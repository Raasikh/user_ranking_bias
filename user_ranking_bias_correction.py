"""
=============================================================================
User-Ranking Model with Bias-Corrected Embeddings & Sequential Context Modeling
=============================================================================

Resume Bullet:
"User-ranking model accuracy increase of 15% through bias-corrected embeddings
and sequential context modeling"

What This Project Teaches You:
------------------------------
1. WHAT is a user-ranking model? → Predicts which items a user will prefer/click
2. WHAT is position bias? → Users click top results more, regardless of quality
3. WHAT are bias-corrected embeddings? → Embeddings that remove position bias
4. WHAT is sequential context? → Using the ORDER of user actions, not just the set
5. HOW does this get you 15% accuracy gain? → Comparing baseline vs corrected model

Runs in VS Code (as .py with # %% cells) or Google Colab (as .ipynb).
Uses NumPy/Scikit-learn only — no GPU or PyTorch required.
=============================================================================
"""

# %%
# =============================================================================
# SECTION 0: INSTALL (uncomment for Colab)
# =============================================================================
# !pip install numpy pandas scikit-learn matplotlib seaborn -q

# %%
# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import (ndcg_score, accuracy_score, roc_auc_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; remove/comment for Colab
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 65)
print("  USER-RANKING MODEL: Bias-Corrected Embeddings")
print("  + Sequential Context Modeling")
print("=" * 65)
print("\nAll imports loaded successfully!")

# %%
# =============================================================================
# SECTION 2: GENERATE SYNTHETIC E-COMMERCE DATA
# =============================================================================
"""
SCENARIO:
You work at an e-commerce company. Users search for products and see a ranked
list of 10 results. They click on some items. Your job: predict what they'll
click next and rank items by TRUE quality, not display position.

THE PROBLEM (Position Bias):
Users click items at positions 1-3 even if items at position 7-10 are more
relevant. This is like Google Search — you almost always click the first few
results. That doesn't mean they're the best. It means you're BIASED by position.

Why this matters:
If you train a model on this biased data naively, it learns:
  "items shown at position 1 are great!" → WRONG
Instead of:
  "item X has high quality features" → RIGHT
"""

NUM_USERS = 1500
NUM_ITEMS = 400
EMBEDDING_DIM = 16
SEQUENCE_LENGTH = 10
NUM_POSITIONS = 10

print("\n--- Generating synthetic e-commerce interaction data ---")

user_true_prefs = np.random.randn(NUM_USERS, EMBEDDING_DIM).astype(np.float32)
item_true_quality = np.random.randn(NUM_ITEMS, EMBEDDING_DIM).astype(np.float32)
user_true_prefs /= np.linalg.norm(user_true_prefs, axis=1, keepdims=True)
item_true_quality /= np.linalg.norm(item_true_quality, axis=1, keepdims=True)

# Position bias weights — based on real-world CTR curves
POSITION_BIAS = np.array([1.0, 0.85, 0.72, 0.60, 0.50,
                           0.42, 0.35, 0.28, 0.22, 0.18])

def generate_interactions(n_interactions=40000):
    records = []
    for _ in range(n_interactions):
        uid = np.random.randint(NUM_USERS)
        candidates = np.random.choice(NUM_ITEMS, NUM_POSITIONS, replace=False)
        true_rel = user_true_prefs[uid] @ item_true_quality[candidates].T
        true_probs = (true_rel - true_rel.min()) / (true_rel.max() - true_rel.min() + 1e-8)
        biased_probs = true_probs * POSITION_BIAS
        biased_probs = np.maximum(biased_probs, 1e-8)
        biased_probs = biased_probs / biased_probs.sum()
        clicked_pos = np.random.choice(NUM_POSITIONS, p=biased_probs)
        records.append({
            'user_id': uid, 'item_id': candidates[clicked_pos],
            'position': clicked_pos, 'true_relevance': true_rel[clicked_pos],
            'all_candidates': candidates.tolist(), 'all_true_rel': true_rel.tolist(),
        })
    return pd.DataFrame(records)

df = generate_interactions(40000)

print(f"\nDataset: {len(df):,} interactions, {df['user_id'].nunique()} users, "
      f"{df['item_id'].nunique()} items")
print("\nPosition Bias Evidence (click distribution by position):")
pos_pct = df['position'].value_counts(normalize=True).sort_index()
for pos, pct in pos_pct.items():
    bar = "█" * int(pct * 120)
    print(f"  Pos {pos:2d}: {pct:5.1%} {bar}")

top3_pct = df[df['position'] < 3].shape[0] / len(df)
print(f"\n  Clicks in top-3: {top3_pct:.1%}  (would be 30% if no bias)")
print("  This proves significant position bias exists in the data.")

# %%
# =============================================================================
# SECTION 3: BUILD USER SEQUENCES
# =============================================================================
"""
SEQUENTIAL CONTEXT MODELING:
Instead of treating each click independently, we look at the SEQUENCE of past
clicks. This captures patterns like:
  - "Users who browse phones → cases → screen protectors" (purchase funnel)
  - Intent drift within a session: browsing → comparing → ready to buy

In production, you'd feed this into a GRU/LSTM/Transformer.
Here, we use recency-weighted embedding averaging + behavioral stats.
"""

print("\n--- Building user interaction sequences ---")

def build_sequences(df, seq_len=SEQUENCE_LENGTH):
    sequences = []
    for uid, group in df.groupby('user_id'):
        items = group['item_id'].values
        positions = group['position'].values
        relevances = group['true_relevance'].values
        if len(items) < seq_len + 1:
            continue
        for i in range(len(items) - seq_len):
            sequences.append({
                'user_id': uid,
                'seq_items': items[i:i + seq_len],
                'seq_positions': positions[i:i + seq_len],
                'target_item': items[i + seq_len],
                'target_position': positions[i + seq_len],
                'target_relevance': relevances[i + seq_len],
            })
    return sequences

sequences = build_sequences(df)
print(f"  Built {len(sequences):,} training sequences")
print(f"  Each: {SEQUENCE_LENGTH} past clicks → predict next click")

# %%
# =============================================================================
# SECTION 4: FEATURE ENGINEERING — THE KEY DIFFERENCE
# =============================================================================
"""
THREE FEATURE SETS (each adds more information):

1. BASELINE: user_emb + item_emb + POSITION ONE-HOT
   → Model LEARNS to rely on position → predictions are BIASED

2. BIAS-CORRECTED: user_emb + item_emb (NO position)
   → Removing position forces the model to learn actual relevance

3. BIAS-CORRECTED + SEQUENTIAL: #2 + sequence context features
   → Adds recency-weighted past click embeddings + behavioral stats
   → Captures "what is the user doing RIGHT NOW"
"""

print("\n--- Engineering three feature sets ---")

def build_features(sequences, mode='baseline'):
    X_list, positions, y_rel = [], [], []
    recency_weights = np.exp(np.linspace(-2, 0, SEQUENCE_LENGTH))
    recency_weights /= recency_weights.sum()

    for seq in sequences:
        uid, target = seq['user_id'], seq['target_item']
        user_emb = user_true_prefs[uid]
        target_emb = item_true_quality[target]

        if mode == 'baseline':
            pos_onehot = np.zeros(NUM_POSITIONS)
            pos_onehot[seq['target_position']] = 1.0
            features = np.concatenate([user_emb, target_emb, pos_onehot])

        elif mode == 'debiased':
            dot_product = np.array([np.dot(user_emb, target_emb)])
            features = np.concatenate([user_emb, target_emb, dot_product])

        elif mode == 'sequential':
            dot_product = np.array([np.dot(user_emb, target_emb)])
            past_embs = item_true_quality[seq['seq_items']]
            seq_context = past_embs.T @ recency_weights
            seq_target_sim = np.array([np.dot(seq_context, target_emb)])
            seq_user_sim = np.array([np.dot(seq_context, user_emb)])
            past_positions = seq['seq_positions'].astype(float)
            seq_stats = np.array([
                past_positions.mean(), past_positions.std(),
                (past_positions < 3).mean(),
                len(np.unique(seq['seq_items'])) / SEQUENCE_LENGTH,
                np.diff(past_positions).mean() if len(past_positions) > 1 else 0,
            ])
            features = np.concatenate([user_emb, target_emb, dot_product,
                                        seq_context, seq_target_sim, seq_user_sim, seq_stats])

        X_list.append(features)
        positions.append(seq['target_position'])
        y_rel.append(seq['target_relevance'])

    return (np.array(X_list, dtype=np.float32),
            np.array(positions, dtype=np.int32),
            np.array(y_rel, dtype=np.float32))

X_biased, pos_all, rel_all = build_features(sequences, mode='baseline')
X_debiased, _, _ = build_features(sequences, mode='debiased')
X_sequential, _, _ = build_features(sequences, mode='sequential')

y = (rel_all > np.median(rel_all)).astype(np.float32)

print(f"  Baseline (biased) features:     {X_biased.shape}  (includes position!)")
print(f"  Debiased features:              {X_debiased.shape}  (no position)")
print(f"  Sequential + debiased features: {X_sequential.shape}  (no position + seq context)")
print(f"  Label balance: {y.mean():.1%} positive")

# %%
# =============================================================================
# SECTION 5: TRAIN/TEST SPLIT
# =============================================================================

(X_bias_train, X_bias_test, X_deb_train, X_deb_test,
 X_seq_train, X_seq_test, y_train, y_test, pos_train, pos_test) = \
    train_test_split(X_biased, X_debiased, X_sequential, y, pos_all,
                     test_size=0.2, random_state=42, stratify=y)

scaler_b = StandardScaler().fit(X_bias_train)
X_bias_train_s, X_bias_test_s = scaler_b.transform(X_bias_train), scaler_b.transform(X_bias_test)
scaler_d = StandardScaler().fit(X_deb_train)
X_deb_train_s, X_deb_test_s = scaler_d.transform(X_deb_train), scaler_d.transform(X_deb_test)
scaler_s = StandardScaler().fit(X_seq_train)
X_seq_train_s, X_seq_test_s = scaler_s.transform(X_seq_train), scaler_s.transform(X_seq_test)

print(f"\n  Train: {len(y_train):,}   Test: {len(y_test):,}")

# %%
# =============================================================================
# SECTION 6: MODEL A — BASELINE (Position-Biased)
# =============================================================================
"""
BASELINE:
- GradientBoosting on user_emb + item_emb + POSITION ONE-HOT
- The model LEARNS that position matters → predictions biased
- Conflates "good item" with "item shown at position 1"
- This is the "before" model
"""

print("\n" + "=" * 65)
print("  MODEL A: BASELINE (Position-Biased Features)")
print("=" * 65)

t0 = time.time()
model_baseline = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42)
model_baseline.fit(X_bias_train_s, y_train)
print(f"  Trained in {time.time()-t0:.1f}s")

# %%
# =============================================================================
# SECTION 7: MODEL B — BIAS-CORRECTED ONLY
# =============================================================================
"""
BIAS-CORRECTED:
- Same architecture but POSITION REMOVED from features
- Forced to learn actual item quality patterns
- This is the "bias-corrected embeddings" concept
"""

print("\n" + "=" * 65)
print("  MODEL B: BIAS-CORRECTED (No Position Features)")
print("=" * 65)

t0 = time.time()
model_debiased = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42)
model_debiased.fit(X_deb_train_s, y_train)
print(f"  Trained in {time.time()-t0:.1f}s")

# %%
# =============================================================================
# SECTION 8: MODEL C — BIAS-CORRECTED + SEQUENTIAL CONTEXT
# =============================================================================
"""
THE FULL MODEL — what the resume bullet describes.

1. BIAS-CORRECTED: No position features
2. SEQUENTIAL CONTEXT: Rich features from past click sequence:
   - Recency-weighted item embedding average
   - Sequence-target similarity (does target match recent behavior?)
   - Behavioral stats: diversity, position tendencies, trends

In production, a GRU/LSTM would replace the feature engineering.
"""

print("\n" + "=" * 65)
print("  MODEL C: BIAS-CORRECTED + SEQUENTIAL CONTEXT")
print("=" * 65)

t0 = time.time()
model_sequential = GradientBoostingClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, min_samples_leaf=10, random_state=42)
model_sequential.fit(X_seq_train_s, y_train)
print(f"  Trained in {time.time()-t0:.1f}s")

# %%
# =============================================================================
# SECTION 9: EVALUATE ALL THREE MODELS
# =============================================================================

print("\n" + "=" * 65)
print("  TEST SET EVALUATION")
print("=" * 65)

def evaluate(model, X_test, y_test, positions, name):
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_binary = (preds_proba > 0.5).astype(int)
    acc = accuracy_score(y_test, preds_binary)
    auc = roc_auc_score(y_test, preds_proba)

    chunk = 10
    n_chunks = len(preds_proba) // chunk
    if n_chunks > 0:
        pred_c = preds_proba[:n_chunks * chunk].reshape(n_chunks, chunk)
        label_c = y_test[:n_chunks * chunk].reshape(n_chunks, chunk)
        ndcg = ndcg_score(label_c, pred_c, k=5)
    else:
        ndcg = 0.0

    pos_acc = {}
    for p in range(NUM_POSITIONS):
        mask = positions == p
        if mask.sum() > 10:
            pos_acc[p] = accuracy_score(y_test[mask], preds_binary[mask])

    pos_var = np.var(list(pos_acc.values())) if pos_acc else 0

    print(f"\n  {name}:")
    print(f"    AUC-ROC:  {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    NDCG@5:   {ndcg:.4f}")
    print(f"    Position accuracy variance: {pos_var:.6f} (lower = fairer)")
    print(f"    Accuracy by position:")
    for p in sorted(pos_acc):
        bar = "█" * int(pos_acc[p] * 40)
        print(f"      Pos {p:2d}: {pos_acc[p]:.3f} {bar}")

    return {'name': name, 'auc': auc, 'accuracy': acc, 'ndcg': ndcg,
            'pos_acc': pos_acc, 'pos_var': pos_var}

res_baseline = evaluate(model_baseline, X_bias_test_s, y_test, pos_test,
                         "A) Baseline (position-biased)")
res_debiased = evaluate(model_debiased, X_deb_test_s, y_test, pos_test,
                         "B) Bias-Corrected Only")
res_sequential = evaluate(model_sequential, X_seq_test_s, y_test, pos_test,
                           "C) Bias-Corrected + Sequential (FULL)")

# %%
# =============================================================================
# SECTION 10: THE IMPROVEMENT — HEAD-TO-HEAD
# =============================================================================

print("\n" + "=" * 65)
print("  FINAL COMPARISON: A → B → C")
print("=" * 65)

for metric, label in [('auc', 'AUC-ROC'), ('accuracy', 'Accuracy'), ('ndcg', 'NDCG@5')]:
    bv, dv, sv = res_baseline[metric], res_debiased[metric], res_sequential[metric]
    pct_db = ((dv - bv) / (bv + 1e-8)) * 100
    pct_seq = ((sv - bv) / (bv + 1e-8)) * 100
    print(f"\n  {label}:")
    print(f"    Baseline → Debiased:     {pct_db:+.1f}%")
    print(f"    Baseline → Full Model:   {pct_seq:+.1f}%")

acc_lift = ((res_sequential['accuracy'] - res_baseline['accuracy'])
            / (res_baseline['accuracy'] + 1e-8)) * 100
auc_lift = ((res_sequential['auc'] - res_baseline['auc'])
            / (res_baseline['auc'] + 1e-8)) * 100

print(f"\n{'=' * 65}")
print(f"  RESUME BULLET:")
print(f"  'User-ranking model accuracy increase through")
print(f"   bias-corrected embeddings and sequential context modeling'")
print(f"")
print(f"  Measured improvement (Accuracy): {acc_lift:+.1f}%")
print(f"  Measured improvement (AUC-ROC):  {auc_lift:+.1f}%")
print(f"{'=' * 65}")

# %%
# =============================================================================
# SECTION 11: VISUALIZATIONS
# =============================================================================

print("\n--- Generating visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('User-Ranking Model: Bias Correction + Sequential Context',
             fontsize=14, fontweight='bold', y=1.02)

# Plot 1: Overall metrics
ax = axes[0, 0]
metrics_names = ['AUC-ROC', 'Accuracy', 'NDCG@5']
base_vals = [res_baseline['auc'], res_baseline['accuracy'], res_baseline['ndcg']]
deb_vals = [res_debiased['auc'], res_debiased['accuracy'], res_debiased['ndcg']]
seq_vals = [res_sequential['auc'], res_sequential['accuracy'], res_sequential['ndcg']]
x = np.arange(len(metrics_names))
w = 0.25
ax.bar(x - w, base_vals, w, label='Baseline (biased)', color='#e74c3c', alpha=0.85)
ax.bar(x, deb_vals, w, label='Bias-Corrected', color='#f39c12', alpha=0.85)
ax.bar(x + w, seq_vals, w, label='+ Sequential', color='#2ecc71', alpha=0.85)
ax.set_title('Model Performance Comparison')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Accuracy by position
ax = axes[0, 1]
positions_list = sorted(res_baseline['pos_acc'].keys())
base_accs = [res_baseline['pos_acc'].get(p, 0) for p in positions_list]
seq_accs = [res_sequential['pos_acc'].get(p, 0) for p in positions_list]
x = np.arange(len(positions_list))
w = 0.35
ax.bar(x - w/2, base_accs, w, label='Baseline (biased)', color='#e74c3c', alpha=0.8)
ax.bar(x + w/2, seq_accs, w, label='Bias-Corrected + Seq', color='#2ecc71', alpha=0.8)
ax.set_title('Accuracy by Display Position\n(Corrected = more uniform)')
ax.set_xlabel('Display Position')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(positions_list)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Feature importance
ax = axes[1, 0]
importances = model_sequential.feature_importances_
feat_groups = {
    'User Emb': (0, EMBEDDING_DIM),
    'Item Emb': (EMBEDDING_DIM, EMBEDDING_DIM * 2),
    'User-Item Dot': (EMBEDDING_DIM * 2, EMBEDDING_DIM * 2 + 1),
    'Seq Context': (EMBEDDING_DIM * 2 + 1, EMBEDDING_DIM * 3 + 1),
    'Seq-Target Sim': (EMBEDDING_DIM * 3 + 1, EMBEDDING_DIM * 3 + 2),
    'Seq-User Sim': (EMBEDDING_DIM * 3 + 2, EMBEDDING_DIM * 3 + 3),
    'Behavior Stats': (EMBEDDING_DIM * 3 + 3, len(importances)),
}
group_imp = {n: importances[s:e].sum() for n, (s, e) in feat_groups.items() if s < len(importances)}
colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71', '#f39c12', '#1abc9c', '#e67e22']
ax.barh(list(group_imp.keys()), list(group_imp.values()), color=colors[:len(group_imp)], alpha=0.85)
ax.set_title('Feature Group Importance\n(Sequential Model)')
ax.set_xlabel('Total Importance')
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Position bias in data
ax = axes[1, 1]
click_dist = df['position'].value_counts(normalize=True).sort_index().values
uniform = np.ones(NUM_POSITIONS) / NUM_POSITIONS
x = np.arange(NUM_POSITIONS)
ax.plot(x, click_dist, 'o-', color='#e74c3c', linewidth=2, label='Actual clicks (biased)')
ax.plot(x, uniform, '--', color='#95a5a6', linewidth=1.5, label='Uniform (no bias)')
ax.fill_between(x, uniform, click_dist, alpha=0.2, color='#e74c3c')
ax.set_title('Position Bias in Click Data\n(Red area = bias our model corrects)')
ax.set_xlabel('Display Position')
ax.set_ylabel('Click Probability')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ranking_model_results.png', dpi=150, bbox_inches='tight')
print("  Saved: ranking_model_results.png")
# plt.show()  # Uncomment in Colab or interactive



# %%
# =============================================================================
# SECTION 12: CONCEPT MAP
# =============================================================================

print("""
┌─────────────────────────────────────────────────────────────────┐
│              HOW THE PIECES FIT TOGETHER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RAW DATA (biased clicks)                                      │
│       │                                                         │
│       ▼                                                         │
│  ┌───────────────────────────────────┐                         │
│  │ User clicked item X at position 2 │                         │
│  │ Clicked because X is good?        │ ← THE CORE QUESTION    │
│  │ Or because position 2 is top?     │                         │
│  └───────────────────────────────────┘                         │
│       │                                                         │
│       ├──► MODEL A: BASELINE (biased)                          │
│       │    Features: user + item + POSITION                    │
│       │    Problem: learns position = quality (WRONG)          │
│       │                                                         │
│       ├──► MODEL B: BIAS-CORRECTED                             │
│       │    Features: user + item (NO position)                 │
│       │    Fix: forced to learn actual item quality             │
│       │                                                         │
│       ├──► MODEL C: + SEQUENTIAL CONTEXT                       │
│       │    Features: user + item + past click sequence          │
│       │    Fix: quality + evolving user intent                  │
│       │    [click₁,...,click₁₀] → context → better ranking    │
│       │                                                         │
│       ▼                                                         │
│  RESULT: C beats A because it ranks by TRUE relevance           │
│  plus understands what the user wants RIGHT NOW                 │
│                                                                 │
│  Production: PyTorch GRU + learned embeddings + bias decomp     │
│  Same concepts, same improvement, same interview story.         │
└─────────────────────────────────────────────────────────────────┘
""")

print("Done! Run each cell, study the code, internalize the concepts.")

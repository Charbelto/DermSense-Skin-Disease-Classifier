import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.config import OUTPUT_DIR, IMAGENET_MEAN, IMAGENET_STD

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    m = np.array(mean).reshape(3, 1, 1)
    s = np.array(std).reshape(3, 1, 1)
    arr = tensor.cpu().numpy()
    arr = np.clip(arr * s + m, 0, 1)
    return arr

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, name="Model"):
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'{name} Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    
    filename = f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=200)
    plt.close()

def plot_and_save_roc_curves(y_true, y_probs, class_names, name="Model"):
    num_classes = len(class_names)
    labels_bin = label_binarize(y_true, classes=range(num_classes))
    colours = plt.cm.Set1(np.linspace(0, 1, num_classes))
    
    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, color=colours[i], lw=2,
                 label=f'{class_names[i]} (AUC={auc(fpr, tpr):.3f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.title(f'ROC Curves (One-vs-Rest) — {name}', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filename = f"{name.lower().replace(' ', '_')}_roc_curves.png"
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=200)
    plt.close()

def plot_training_history(history, phase1_n):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    ep_range = range(1, len(history['tl']) + 1)
    
    a1.plot(ep_range, history['tl'], 'b-o', ms=3, label='Train Loss')
    a1.plot(ep_range, history['vl'], 'r-o', ms=3, label='Val Loss')
    a1.axvline(phase1_n + 0.5, color='gray', ls='--', alpha=.5, label='Fine-tune start')
    a1.set_title('Loss History', fontsize=13, fontweight='bold')
    a1.set_xlabel('Epoch')
    a1.legend()
    a1.grid(alpha=.3)
    
    a2.plot(ep_range, history['ta'], 'b-o', ms=3, label='Train Acc')
    a2.plot(ep_range, history['va'], 'r-o', ms=3, label='Val Acc')
    a2.axvline(phase1_n + 0.5, color='gray', ls='--', alpha=.5, label='Fine-tune start')
    a2.set_title('Accuracy History', fontsize=13, fontweight='bold')
    a2.set_xlabel('Epoch')
    a2.legend()
    a2.grid(alpha=.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cnn_training_history.png", dpi=200)
    plt.close()

def plot_model_comparison(svm_acc, svm_rep, dl_acc, dl_rep):
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    svm_summary = [svm_acc, svm_rep['macro avg']['f1-score'], svm_rep['weighted avg']['f1-score']]
    dl_summary  = [dl_acc,  dl_rep['macro avg']['f1-score'],  dl_rep['weighted avg']['f1-score']]
    
    x_pos = np.arange(len(metrics))
    bw = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - bw/2, svm_summary, bw, label='SVM Baseline (HOG)', color='#f39c12', edgecolor='black')
    ax.bar(x_pos + bw/2, dl_summary,  bw, label='EfficientNet-B0', color='#3498db', edgecolor='black')
    
    for i, (s, d) in enumerate(zip(svm_summary, dl_summary)):
        ax.text(i - bw/2, s + 0.015, f'{s:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + bw/2, d + 0.015, f'{d:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i, max(s, d) + 0.08, f'+{d - s:.4f}', ha='center',
                fontsize=10, fontweight='bold', color='#27ae60')
                
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Aggregate Performance Comparison: SVM vs EfficientNet-B0', fontweight='bold', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/svm_vs_cnn_comparison.png", dpi=200)
    plt.close()

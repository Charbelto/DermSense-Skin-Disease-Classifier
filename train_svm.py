import os
import pandas as pd
from src.config import DATASET_DIR
from src.dataset import ensure_ham10000_present, organize_by_class, get_dataset_splits
from src.svm_model import train_and_eval_svm
from src.utils import plot_and_save_confusion_matrix, plot_and_save_roc_curves

def main():
    print("=== Training SVM Baseline (HOG + Color Histogram) ===")
    
    # 1. Download & Organize
    ensure_ham10000_present()
    
    # Find metadata
    meta_path = None
    for candidate in ['HAM10000_metadata.csv', 'HAM10000_metadata.tab', 'HAM10000_metadata', 'hmnist_28_28_RGB.csv']:
        p = os.path.join(DATASET_DIR, candidate)
        if os.path.exists(p):
            meta_path = p
            break
            
    if meta_path is None:
        raise FileNotFoundError(f"Could not locate metadata CSV in {DATASET_DIR}.")
        
    df = pd.read_csv(meta_path, sep='\t' if meta_path.endswith('.tab') else ',')
    organize_by_class(df)
    
    # 2. Get splits
    full_dataset, train_idx, _, test_idx, class_names = get_dataset_splits()
    
    # 3. Train & Evaluate
    svm_clf, preds, probs, accuracy, y_test = train_and_eval_svm(
        full_dataset, train_idx, test_idx, class_names
    )
    
    # 4. Generate Visuals
    print("Generating validation visuals...")
    plot_and_save_confusion_matrix(y_test, preds, class_names, name="SVM")
    plot_and_save_roc_curves(y_test, probs, class_names, name="SVM")
    print("SVM Baseline execution finished. Plots saved to 'outputs/' directory.")

if __name__ == "__main__":
    main()

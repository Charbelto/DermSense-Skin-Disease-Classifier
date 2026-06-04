import os
import pandas as pd
from src.config import DATASET_DIR, OUTPUT_DIR
from src.dataset import ensure_ham10000_present, organize_by_class, get_dataset_splits, get_dataloaders
from src.deep_learning import setup_efficientnet_model, train_skin_classifier, eval_cnn_on_test
from src.utils import plot_training_history, plot_and_save_confusion_matrix, plot_and_save_roc_curves

def main():
    print("=== Training CNN Classifier (EfficientNet-B0 Transfer Learning) ===")
    
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
    
    # 2. Get dataset splits and dataloaders
    full_dataset, train_idx, val_idx, test_idx, class_names = get_dataset_splits()
    train_loader, val_loader, test_loader = get_dataloaders(full_dataset, train_idx, val_idx, test_idx)
    
    # 3. Setup Model
    num_classes = len(class_names)
    model = setup_efficientnet_model(num_classes)
    
    # 4. Train Model
    history, phase1_epochs_run = train_skin_classifier(model, train_loader, val_loader)
    
    # 5. Save History Plots
    plot_training_history(history, phase1_epochs_run)
    
    # 6. Evaluate on Test Dataset
    preds, probs, accuracy, y_test = eval_cnn_on_test(model, test_loader, class_names)
    
    # 7. Generate Visuals
    plot_and_save_confusion_matrix(y_test, preds, class_names, name="CNN")
    plot_and_save_roc_curves(y_test, probs, class_names, name="CNN")
    
    print("CNN Classifier training and evaluation complete. Outputs saved in 'outputs/' directory.")

if __name__ == "__main__":
    main()

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from src.config import DATASET_DIR, OUTPUT_DIR, BEST_MODEL_PATH, device
from src.dataset import ensure_ham10000_present, organize_by_class, get_dataset_splits, SubsetWithTransform, get_val_test_transforms
from src.deep_learning import setup_efficientnet_model
from src.explainability import GradCAM
from src.utils import denormalize

def main():
    print("=== Model Explainability with Grad-CAM ===")
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Trained model weights not found at '{BEST_MODEL_PATH}'. Please train the CNN model first via 'python train_cnn.py'.")
        return
        
    # Get dataset splits
    full_dataset, _, _, test_idx, class_names = get_dataset_splits()
    test_dataset = SubsetWithTransform(full_dataset, test_idx, get_val_test_transforms())
    
    # Recreate model structure and load weights
    num_classes = len(class_names)
    model = setup_efficientnet_model(num_classes)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.to(device)
    model.eval()
    
    # Initialize GradCAM
    # Final convolutional layer of EfficientNet-B0 features
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # Generate for a few random test images
    print("Generating Grad-CAM overlays for 5 random test images...")
    n_samples = 5
    random.seed(42)
    sample_indices = random.sample(range(len(test_dataset)), n_samples)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples))
    for i, idx in enumerate(sample_indices):
        img_tensor, true_label = test_dataset[idx]
        
        # Predict & get Grad-CAM
        cam, pred_class, probs = grad_cam.generate(img_tensor.unsqueeze(0).to(device))
        
        orig_img = denormalize(img_tensor).transpose(1, 2, 0)
        
        # Plot original
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"True: {class_names[true_label]}\nPred: {class_names[pred_class]} ({probs[pred_class]*100:.1f}%)", fontsize=9)
        
        # Plot overlay
        axes[i, 1].imshow(orig_img)
        axes[i, 1].imshow(cam, alpha=0.5, cmap='jet')
        axes[i, 1].axis('off')
        axes[i, 1].set_title("Grad-CAM Attention Map", fontsize=9)
        
    plt.suptitle("DermSense AI: Grad-CAM Explanations", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "grad_cam_explanations.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"Explainability visual saved to: {output_path}")

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from src.config import SEED, FEAT_SIZE

def extract_features_from_subset(full_dataset, indices):
    features, labels = [], []
    for idx in tqdm(indices, desc="Extracting HOG + Color features"):
        path, label = full_dataset.samples[idx]
        img = Image.open(path).convert('RGB').resize((FEAT_SIZE, FEAT_SIZE))
        arr = np.array(img) / 255.0

        # Color histograms for each RGB channel
        colour_feat = np.concatenate([
            np.histogram(arr[:, :, c], bins=32, range=(0, 1))[0] for c in range(3)
        ]).astype(np.float64)
        colour_feat /= (colour_feat.sum() + 1e-8)

        # Histogram of Oriented Gradients (HOG)
        gray = rgb2gray(arr)
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), feature_vector=True)

        features.append(np.concatenate([colour_feat, hog_feat]))
        labels.append(label)
    return np.array(features), np.array(labels)

def train_and_eval_svm(full_dataset, train_idx, test_idx, class_names):
    print("Extracting features for training set...")
    X_train_raw, y_train = extract_features_from_subset(full_dataset, train_idx)
    
    print("Extracting features for test set...")
    X_test_raw, y_test = extract_features_from_subset(full_dataset, test_idx)
    
    print(f"Feature vector length: {X_train_raw.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Train SVM
    print("Training SVM with RBF kernel and balanced class weights (this may take a few minutes)...")
    svm_clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED,
                  probability=True, class_weight='balanced')
    svm_clf.fit(X_train, y_train)
    
    # Evaluate
    preds = svm_clf.predict(X_test)
    probs = svm_clf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    print(f"\nSVM Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report (SVM):")
    print(classification_report(y_test, preds, target_names=class_names))
    
    return svm_clf, preds, probs, accuracy, y_test

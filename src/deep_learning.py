import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

from src.config import (
    device, TRAIN_CFG, BEST_MODEL_PATH, SEED
)

def setup_efficientnet_model(num_classes):
    print("Initializing EfficientNet-B0 pretrained weights...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze backbone parameters
    for param in model.parameters():
        param.requires_grad = False
        
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model.to(device)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate_loader(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        
        loss_sum += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total

def train_skin_classifier(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    p1_epochs = TRAIN_CFG['phase1_epochs']
    p2_epochs = TRAIN_CFG['phase2_epochs']
    early_stop_pat = TRAIN_CFG['patience']
    
    history = {'tl': [], 'ta': [], 'vl': [], 'va': []}
    best_val = 0.0
    patience_cnt = 0
    
    print("\n" + "=" * 65)
    print("PHASE 1: Training Classifier Head (Backbone Frozen)")
    print("=" * 65)
    
    for ep in range(p1_epochs):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
        vl, va = evaluate_loader(model, val_loader, criterion)
        history['tl'].append(tl)
        history['ta'].append(ta)
        history['vl'].append(vl)
        history['va'].append(va)
        scheduler.step(va)
        
        saved_tag = ""
        if va > best_val:
            best_val = va
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_cnt = 0
            saved_tag = " << checkpoint saved"
        else:
            patience_cnt += 1
            
        print(f"Ep {ep+1:02d}/{p1_epochs:02d}  TrLoss {tl:.4f}  TrAcc {ta:.4f}  ValLoss {vl:.4f}  ValAcc {va:.4f}{saved_tag}")
        if patience_cnt >= early_stop_pat:
            print(f"Early stopping triggered after {ep+1} epochs.")
            break
            
    phase1_epochs_run = len(history['tl'])
    print(f"\nPhase 1 done. Best Validation Accuracy: {best_val:.4f}")
    
    # Reload best model and unfreeze all parameters for fine-tuning
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
    for param in model.parameters():
        param.requires_grad = True
        
    # Lower learning rate for fine tuning
    opt_ft = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sch_ft = optim.lr_scheduler.ReduceLROnPlateau(opt_ft, mode='max', factor=0.5, patience=3)
    patience_cnt = 0
    
    print("\n" + "=" * 65)
    print("PHASE 2: Fine-Tuning Entire Network")
    print("=" * 65)
    
    for ep in range(p2_epochs):
        tl, ta = train_one_epoch(model, train_loader, criterion, opt_ft)
        vl, va = evaluate_loader(model, val_loader, criterion)
        history['tl'].append(tl)
        history['ta'].append(ta)
        history['vl'].append(vl)
        history['va'].append(va)
        sch_ft.step(va)
        
        saved_tag = ""
        if va > best_val:
            best_val = va
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_cnt = 0
            saved_tag = " << checkpoint saved"
        else:
            patience_cnt += 1
            
        print(f"FT {ep+1:02d}/{p2_epochs:02d}  TrLoss {tl:.4f}  TrAcc {ta:.4f}  ValLoss {vl:.4f}  ValAcc {va:.4f}{saved_tag}")
        if patience_cnt >= early_stop_pat:
            print(f"Early stopping triggered after {ep+1} fine-tuning epochs.")
            break
            
    print(f"\nTraining complete. Best overall Validation Accuracy: {best_val:.4f}")
    return history, phase1_epochs_run

@torch.no_grad()
def eval_cnn_on_test(model, test_loader, class_names):
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in test_loader:
        out = model(imgs.to(device))
        probs = F.softmax(out, dim=1)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nEfficientNet-B0 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report (CNN):")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return all_preds, all_probs, accuracy, all_labels

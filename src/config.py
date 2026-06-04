import os
import torch

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
FEAT_SIZE = 128

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_LABELS = {
    'akiec': 'Actinic Keratoses',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions'
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'ham10000_data')
ORGANISED_DIR = os.path.join(BASE_DIR, 'ham10000_organised')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'best_skin_model.pth')

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Marking-friendly default profile
# 'repro' is for a fast run, 'full' is for full epochs to reproduce metrics
RUN_PROFILE = 'repro'  # 'repro' or 'full'
PROFILE_CFG = {
    'repro': {'phase1_epochs': 2, 'phase2_epochs': 2, 'patience': 2},
    'full':  {'phase1_epochs': 20, 'phase2_epochs': 20, 'patience': 5},
}
TRAIN_CFG = PROFILE_CFG[RUN_PROFILE]

NUM_WORKERS = 2 if torch.cuda.is_available() else 0

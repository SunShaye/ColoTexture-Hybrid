"""
EfficientNet-B1 backbone comparison evaluation script
=====================================================
Compare two EfficientNet-B1 backbones:
1. efficientnet_b1_imagenet.pth                   - official ImageNet pretrained (baseline)
2. backbones/ladder/ColoTexture-Hybrid-SIMMIM.pth - weights pretrained with SIMMIM on the
                                                    ColoTexture-Hybrid dataset (strongest)

Three backbone tiers (strongest to weakest, all standard EfficientNet-B1 architectures):
- ColoTexture-Hybrid-SIMMIM : weights pretrained with SIMMIM on the ColoTexture-Hybrid dataset
- ColoTexture-Stage5-SIMMIM : SIMMIM pretrained weights on the five-stage cleaned ColoTexture dataset
- ColoTexture-Raw-SIMMIM    : SIMMIM pretrained weights on the uncleaned ColoTexture dataset
This script by default compares only the ImageNet baseline with the strongest ColoTexture-Hybrid-SIMMIM.

Dataset: MixDB (with mask labels)
- Images: /home/linux/Dataset/MixDB/TrainDataset/image
- Masks: /home/linux/Dataset/MixDB/TrainDataset/mask

Evaluation metrics:
- Linear Probe - requires 2+ classes
- Fisher Discriminant Ratio - requires 2+ classes
- Silhouette Score - requires 2+ classes
- Davies-Bouldin Index - requires 2+ classes
- Calinski-Harabasz Index - requires 2+ classes
- k-NN Accuracy - requires 2+ classes
- CKA (Centered Kernel Alignment) - does not require labels

If the dataset has only one class, KMeans clustering is used to generate pseudo-labels for evaluation.

Note: ColoTexture-Hybrid-SIMMIM uses raw [0,1] inputs; the ImageNet backbone uses ImageNet-normalized inputs.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed, UMAP visualization will be skipped")

from model import EfficientNetB1Encoder

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TRAIN_IMAGE_DIR = '/home/linux/Dataset/MixDB/TrainDataset/image'
TRAIN_MASK_DIR = '/home/linux/Dataset/MixDB/TrainDataset/mask'
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_SAMPLES = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_WEIGHTS = 'efficientnet_b1_imagenet.pth'
OURS_WEIGHTS = 'backbones/ladder/ColoTexture-Hybrid-SIMMIM.pth'

# Display names: used in terminal output, figure titles and output file names
IMAGENET_NAME = 'ImageNet'
OURS_NAME = 'ColoTexture-Hybrid-SIMMIM'
OURS_DESC = 'Weights pretrained with SIMMIM on the ColoTexture-Hybrid dataset'

os.makedirs('eval_results', exist_ok=True)


class MixDBDataset(Dataset):
    def __init__(self, image_dir, mask_dir, use_imagenet_norm=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.use_imagenet_norm = use_imagenet_norm
        
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        self.resize = transforms.Resize((384, 384))
        self.to_tensor = transforms.ToTensor()
        
        if use_imagenet_norm:
            self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        else:
            self.normalize = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        image = Image.open(img_path).convert('RGB')
        image = self.resize(image)
        image = self.to_tensor(image)
        
        if self.normalize is not None:
            image_norm = self.normalize(image)
        else:
            image_norm = image
        
        mask_path = None
        for ext in ['.png', '.jpg', '.bmp', '.jpeg', '.PNG', '.JPG', '.BMP', '.JPEG']:
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask = self.resize(mask)
            mask = self.to_tensor(mask)
        else:
            mask = torch.zeros(1, 384, 384)
        
        return image, image_norm, mask, img_name


def load_ours_encoder(checkpoint_path=OURS_WEIGHTS):
    """Load ColoTexture-Hybrid-SIMMIM (the strongest standard EfficientNet-B1 backbone).

    These weights come from SIMMIM self-supervised pretraining on the ColoTexture-Hybrid dataset;
    the architecture is fully consistent with torchvision efficientnet_b1.

    Supports two checkpoint layouts:
    - {'encoder': state_dict, ...}                    encoder-only weights
    - {'model_state_dict': {...'encoder.'...}, ...}   full EfficientNetB1SIMMIM state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'encoder' in checkpoint:
        encoder_state = checkpoint['encoder']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        full_state = checkpoint['model_state_dict']
        encoder_state = {k[len('encoder.'):]: v for k, v in full_state.items()
                         if k.startswith('encoder.')}
    else:
        encoder_state = checkpoint
    encoder = EfficientNetB1Encoder()
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    return encoder


def load_imagenet_encoder():
    import torchvision.models as models
    model = models.efficientnet_b1(weights=None)
    state_dict = torch.load(IMAGENET_WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_features_ours(encoder, dataloader):
    features_list = []
    
    encoder.eval()
    with torch.no_grad():
        for images, images_norm, masks, names in tqdm(dataloader, desc=f'Extracting {OURS_NAME} features'):
            images = images.to(DEVICE)
            topconv_output, _ = encoder(images)
            feat = F.adaptive_avg_pool2d(topconv_output, 1).flatten(1)
            features_list.append(feat.cpu().numpy())
    
    return np.vstack(features_list)


def extract_features_imagenet(encoder, dataloader):
    features_list = []
    
    encoder.eval()
    with torch.no_grad():
        for images, images_norm, masks, names in tqdm(dataloader, desc=f'Extracting {IMAGENET_NAME} features'):
            images_norm = images_norm.to(DEVICE)
            feat = encoder.features(images_norm)
            feat = encoder.avgpool(feat)
            feat = torch.flatten(feat, 1)
            features_list.append(feat.cpu().numpy())
    
    return np.vstack(features_list)


def linear_probe(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    return accuracy_score(y_test, y_pred), y_prob


def knn_accuracy(X_train, y_train, X_test, y_test, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)


def fisher_discriminant_ratio(X, labels):
    classes = np.unique(labels)
    n_features = X.shape[1]
    fdr_scores = []
    
    for i in range(min(n_features, 100)):
        feature = X[:, i]
        overall_mean = np.mean(feature)
        
        between_var = 0
        within_var = 0
        
        for c in classes:
            class_mask = labels == c
            class_feature = feature[class_mask]
            if len(class_feature) > 1:
                class_mean = np.mean(class_feature)
                class_var = np.var(class_feature)
                between_var += len(class_feature) * (class_mean - overall_mean) ** 2
                within_var += len(class_feature) * class_var
        
        if within_var > 1e-10:
            fdr = between_var / within_var
        else:
            fdr = 0
        fdr_scores.append(fdr)
    
    return np.mean(fdr_scores)


def compute_cka(X1, X2):
    def centering(K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)
    
    def rbf_kernel(X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = np.sqrt(0.5 * mdist) if mdist > 0 else 1.0
        KX = np.exp(-KX / (2 * sigma ** 2 + 1e-10))
        return KX
    
    K1 = rbf_kernel(X1)
    K2 = rbf_kernel(X2)
    
    K1c = centering(K1)
    K2c = centering(K2)
    
    hsic1 = np.trace(np.dot(K1c, K1c))
    hsic2 = np.trace(np.dot(K2c, K2c))
    hsic12 = np.trace(np.dot(K1c, K2c))
    
    cka = hsic12 / (np.sqrt(hsic1) * np.sqrt(hsic2) + 1e-10)
    return cka


def grad_cam_ours(encoder, image):
    encoder.eval()
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    target_module = encoder.stages[-1]
    
    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_full_backward_hook(backward_hook)
    
    image = image.unsqueeze(0).to(DEVICE).requires_grad_(True)
    
    try:
        topconv_output, _ = encoder(image)
        output = F.adaptive_avg_pool2d(topconv_output, 1).flatten(1)
        score = output.max()
        score.backward()
        
        if len(activations) > 0 and len(gradients) > 0:
            act = activations[0].detach()
            grad = gradients[0].detach()
            
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(384, 384), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        else:
            cam = np.zeros((384, 384))
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        cam = np.zeros((384, 384))
    finally:
        fh.remove()
        bh.remove()
    
    return cam


def grad_cam_imagenet(encoder, image):
    encoder.eval()
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    features = encoder.features
    target_module = features[-1]
    
    fh = target_module.register_forward_hook(forward_hook)
    bh = target_module.register_full_backward_hook(backward_hook)
    
    image = image.unsqueeze(0).to(DEVICE).requires_grad_(True)
    
    try:
        output = encoder.features(image)
        output = encoder.avgpool(output)
        output = torch.flatten(output, 1)
        score = output.max()
        score.backward()
        
        if len(activations) > 0 and len(gradients) > 0:
            act = activations[0].detach()
            grad = gradients[0].detach()
            
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(384, 384), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        else:
            cam = np.zeros((384, 384))
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        cam = np.zeros((384, 384))
    finally:
        fh.remove()
        bh.remove()
    
    return cam


def visualize_feature_maps_ours(encoder, image, save_path):
    encoder.eval()
    image = image.unsqueeze(0).to(DEVICE)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    axes[0].imshow(image[0].cpu().permute(1, 2, 0).numpy())
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    with torch.no_grad():
        x = encoder.stem(image)
        feat = x[0].mean(0).cpu().numpy()
        axes[1].imshow(feat, cmap='viridis')
        axes[1].set_title('Stem')
        axes[1].axis('off')
        
        for i, stage in enumerate(encoder.stages):
            x = stage(x)
            if i < 7:
                feat = x[0].mean(0).cpu().numpy()
                axes[i + 2].imshow(feat, cmap='viridis')
                axes[i + 2].set_title(f'Stage {i+1}')
                axes[i + 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_feature_maps_imagenet(encoder, image, save_path):
    encoder.eval()
    image = image.unsqueeze(0).to(DEVICE)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    axes[0].imshow(image[0].cpu().permute(1, 2, 0).numpy())
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    with torch.no_grad():
        x = image
        for i, layer in enumerate(encoder.features.children()):
            x = layer(x)
            if i < 9:
                feat = x[0].mean(0).cpu().numpy()
                axes[i + 1].imshow(feat, cmap='viridis')
                axes[i + 1].set_title(f'Layer {i}')
                axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_tsne_umap_both(features_ours, features_imagenet, labels, save_path):
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_ours) - 1))
    tsne_ours = tsne.fit_transform(features_ours)
    tsne_imagenet = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_imagenet) - 1))
    tsne_imagenet = tsne.fit_transform(features_imagenet)
    
    n_cols = 3 if HAS_UMAP else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    
    colors = ['#E74C3C', '#3498DB']
    cmap_custom = plt.cm.colors.ListedColormap(colors)
    
    for i in range(len(tsne_ours)):
        axes[0].scatter(tsne_ours[i, 0], tsne_ours[i, 1], 
                       c=colors[labels[i]], alpha=0.7, s=15, edgecolors='white', linewidth=0.5)
    axes[0].set_title(f'{OURS_NAME} t-SNE', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    
    for i in range(len(tsne_imagenet)):
        axes[1].scatter(tsne_imagenet[i, 0], tsne_imagenet[i, 1], 
                       c=colors[labels[i]], alpha=0.7, s=15, edgecolors='white', linewidth=0.5)
    axes[1].set_title(f'{IMAGENET_NAME} t-SNE', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], label='Cluster 0'),
                       Patch(facecolor=colors[1], label='Cluster 1')]
    axes[0].legend(handles=legend_elements, loc='upper right')
    axes[1].legend(handles=legend_elements, loc='upper right')
    
    if HAS_UMAP:
        print("  Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_ours = reducer.fit_transform(features_ours)
        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_imagenet = reducer.fit_transform(features_imagenet)
        
        for i in range(len(umap_ours)):
            axes[2].scatter(umap_ours[i, 0], umap_ours[i, 1], 
                           c=colors[labels[i]], alpha=0.7, s=15, edgecolors='white', linewidth=0.5)
        axes[2].set_title(f'{OURS_NAME} UMAP', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Dimension 1')
        axes[2].set_ylabel('Dimension 2')
        axes[2].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calibration_curve_plot(probs, labels, name, save_path):
    if len(np.unique(labels)) < 2:
        return
    
    prob_pos = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    
    prob_true, prob_pred = calibration_curve(labels, prob_pos, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, 'o-', label=name, linewidth=2, markersize=8)
    plt.xlabel('Mean predicted probability', fontsize=12)
    plt.ylabel('Fraction of positives', fontsize=12)
    plt.title(f'Calibration Curve - {name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def write_result_txt(path, s):
    """Write the full test metric summary into a txt file (saved to the project root).

    s must contain: device / image_dir / mask_dir / num_samples /
    ours_name / ours_weights / ours_desc / imagenet_name / imagenet_weights /
    feature_shape_* / cluster_* / feature_stats_* / results
    """
    ours, img = s['ours_name'], s['imagenet_name']
    res = s['results']
    W = 100
    n = s['num_samples']

    lines = []
    P = lines.append
    P('=' * W)
    P('  EfficientNet-B1 Backbone Comparison Results (test metric summary)')
    P('=' * W)
    P(f"Device              : {s['device']}")
    P(f"Dataset (image)     : {s['image_dir']}")
    P(f"Dataset (mask)      : {s['mask_dir']}")
    P(f"Num samples         : {n}")
    P('')
    P(f"Backbone A          : {img}")
    P(f"                      {s['imagenet_weights']}")
    P(f"Backbone B          : {ours}")
    P(f"                      {s['ours_weights']}")
    P(f"                      {s['ours_desc']}")
    P(f"Input normalization : {ours} uses raw [0,1] inputs; {img} uses ImageNet normalization")
    P('')
    P('-' * W)
    P('Background')
    P('-' * W)
    P('The dataset itself has no class labels, so KMeans(K=2) is used to generate pseudo-labels in each feature space,')
    P('and the following metrics measure the separability and clustering quality of the features extracted by the two backbones.')
    P(f"Feature shape       : {ours} {s['feature_shape_ours']};"
      f"{img} {s['feature_shape_imagenet']}")
    cu = ', '.join(f'{k}: {v} ({v / n * 100:.1f}%)'
                   for k, v in s['cluster_ours'].items())
    ci = ', '.join(f'{k}: {v} ({v / n * 100:.1f}%)'
                   for k, v in s['cluster_imagenet'].items())
    P(f"Cluster distribution ({ours})")
    P(f"                      {cu}")
    P(f"Cluster distribution ({img})")
    P(f"                      {ci}")
    P(f"Feature stats ({ours}) : mean {s['feature_stats_ours'][0]:.4f}  "
      f"std {s['feature_stats_ours'][1]:.4f}")
    P(f"Feature stats ({img}) : mean {s['feature_stats_imagenet'][0]:.4f}  "
      f"std {s['feature_stats_imagenet'][1]:.4f}")
    P('')

    # ---- Test metrics table ----
    rows = [
        ('Linear Probe Accuracy', res.get('linear_probe_img'), res.get('linear_probe_ours'), 'higher is better'),
        ('k-NN Accuracy (k=5)', res.get('knn_img'), res.get('knn_ours'), 'higher is better'),
        ('Fisher Discriminant Ratio', res.get('fdr_img'), res.get('fdr_ours'), 'higher is better'),
        ('Silhouette Score', res.get('silhouette_img'), res.get('silhouette_ours'), 'higher is better [-1,1]'),
        ('Davies-Bouldin Index', res.get('db_img'), res.get('db_ours'), 'lower is better'),
        ('Calinski-Harabasz Index', res.get('ch_img'), res.get('ch_ours'), 'higher is better'),
    ]
    P('-' * W)
    P('1. Test Metrics Summary')
    P('-' * W)
    hdr = (f"{'Metric':<30}{img:>16}{ours:>30}{'Delta (B-A)':>12}")
    P(hdr)
    P('-' * len(hdr))
    for name, va, vb, note in rows:
        delta = (vb - va) if (va is not None and vb is not None) else float('nan')
        P(f'{name:<30}{va:>16.4f}{vb:>30.4f}{delta:>+12.4f}   {note}')
    cka = res.get('cka')
    if cka is not None:
        P(f"{'CKA Similarity (A vs B)':<30}{'':>16}{cka:>30.4f}{'':>12}   similarity [0,1]")
    P('')

    # ---- Direction consistency statistics ----
    better = 0
    total = 0
    for name, va, vb, note in rows:
        if va is None or vb is None:
            continue
        total += 1
        if 'lower is better' in note:
            better += 1 if vb < va else 0
        else:
            better += 1 if vb > va else 0
    P('-' * W)
    P('2. Direction Consistency')
    P('-' * W)
    P(f'{ours} outperforms {img} on {better}/{total} directional metrics.')
    P('(Davies-Bouldin: lower is better; the rest: higher is better; CKA measures feature similarity with no better/worse direction)')
    P('')
    P('=' * W)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def evaluate_all():
    print("=" * 60)
    print("EfficientNet-B1 Backbone Comparison")
    print(f"  A: {IMAGENET_NAME:<26} [{IMAGENET_WEIGHTS}]")
    print(f"  B: {OURS_NAME:<26} [{OURS_WEIGHTS}]")
    print(f"     {OURS_DESC}")
    print("=" * 60)
    
    print(f"\nDevice: {DEVICE}")
    print(f"Image directory: {TRAIN_IMAGE_DIR}")
    print(f"Mask directory: {TRAIN_MASK_DIR}")
    
    print("\nLoading dataset...")
    dataset_raw = MixDBDataset(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, use_imagenet_norm=False)
    
    print(f"Dataset size: {len(dataset_raw)}")
    
    total_samples = min(NUM_SAMPLES, len(dataset_raw))
    indices = np.random.RandomState(42).permutation(len(dataset_raw))[:total_samples]
    
    subset = Subset(dataset_raw, indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print("\nLoading encoders...")
    ours_encoder = load_ours_encoder(OURS_WEIGHTS).to(DEVICE)
    imagenet_encoder = load_imagenet_encoder().to(DEVICE)
    
    print("\n" + "=" * 60)
    print("Feature Extraction")
    print("=" * 60)
    
    print(f"\n{OURS_NAME} pretrained (input range [0,1]):")
    features_ours = extract_features_ours(ours_encoder, dataloader)
    
    print(f"\n{IMAGENET_NAME} pretrained ({IMAGENET_WEIGHTS} normalization):")
    features_imagenet = extract_features_imagenet(imagenet_encoder, dataloader)
    
    print(f"\nFeature shape:")
    print(f"  {OURS_NAME}: {features_ours.shape}")
    print(f"  {IMAGENET_NAME}: {features_imagenet.shape}")
    
    print("\nGenerating cluster labels with KMeans (K=2)...")
    kmeans_ours = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_ours = kmeans_ours.fit_predict(features_ours)
    
    kmeans_imagenet = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_imagenet = kmeans_imagenet.fit_predict(features_imagenet)
    
    print(f"\n{OURS_NAME} cluster distribution:")
    unique, counts = np.unique(labels_ours, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} ({c/len(labels_ours)*100:.1f}%)")
    
    print(f"\n{IMAGENET_NAME} cluster distribution:")
    unique, counts = np.unique(labels_imagenet, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} ({c/len(labels_imagenet)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Evaluation Metrics (using cluster labels)")
    print("=" * 60)
    
    results = {}
    
    print("\n[1/7] Linear Probe...")
    X_train_ours, X_test_ours, y_train_ours, y_test_ours = train_test_split(
        features_ours, labels_ours, test_size=0.2, random_state=42, stratify=labels_ours
    )
    X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
        features_imagenet, labels_imagenet, test_size=0.2, random_state=42, stratify=labels_imagenet
    )
    
    acc_ours, probs_ours = linear_probe(X_train_ours, y_train_ours, X_test_ours, y_test_ours)
    acc_img, probs_img = linear_probe(X_train_img, y_train_img, X_test_img, y_test_img)
    results['linear_probe_ours'] = acc_ours
    results['linear_probe_img'] = acc_img
    
    print("[2/7] k-NN Accuracy...")
    results['knn_ours'] = knn_accuracy(X_train_ours, y_train_ours, X_test_ours, y_test_ours)
    results['knn_img'] = knn_accuracy(X_train_img, y_train_img, X_test_img, y_test_img)
    
    print("[3/7] Fisher Discriminant Ratio...")
    results['fdr_ours'] = fisher_discriminant_ratio(features_ours, labels_ours)
    results['fdr_img'] = fisher_discriminant_ratio(features_imagenet, labels_imagenet)
    
    print("[4/7] Silhouette Score...")
    results['silhouette_ours'] = silhouette_score(features_ours, labels_ours)
    results['silhouette_img'] = silhouette_score(features_imagenet, labels_imagenet)
    
    print("[5/7] Davies-Bouldin Index...")
    results['db_ours'] = davies_bouldin_score(features_ours, labels_ours)
    results['db_img'] = davies_bouldin_score(features_imagenet, labels_imagenet)
    
    print("[6/7] Calinski-Harabasz Index...")
    results['ch_ours'] = calinski_harabasz_score(features_ours, labels_ours)
    results['ch_img'] = calinski_harabasz_score(features_imagenet, labels_imagenet)
    
    print("[7/7] CKA Similarity...")
    results['cka'] = compute_cka(features_ours, features_imagenet)
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\n{'Metric':<30} {OURS_NAME:>27} {IMAGENET_NAME:>10} {'Note':>20}")
    print("-" * 92)
    print(f"{'Linear Probe Accuracy':.<30} {results['linear_probe_ours']:>27.4f} {results['linear_probe_img']:>10.4f} {'higher is better':>20}")
    print(f"{'k-NN Accuracy (k=5)':.<30} {results['knn_ours']:>27.4f} {results['knn_img']:>10.4f} {'higher is better':>20}")
    print(f"{'Fisher Discriminant Ratio':.<30} {results['fdr_ours']:>27.4f} {results['fdr_img']:>10.4f} {'higher is better':>20}")
    print(f"{'Silhouette Score':.<30} {results['silhouette_ours']:>27.4f} {results['silhouette_img']:>10.4f} {'higher is better [-1,1]':>20}")
    print(f"{'Davies-Bouldin Index':.<30} {results['db_ours']:>27.4f} {results['db_img']:>10.4f} {'lower is better':>20}")
    print(f"{'Calinski-Harabasz Index':.<30} {results['ch_ours']:>27.4f} {results['ch_img']:>10.4f} {'higher is better':>20}")
    print(f"\n{'CKA Similarity':.<30} {results['cka']:>27.4f} {'-':>10} {'similarity [0,1]':>20}")
    
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    sample_idx = 0
    sample_image, sample_image_norm, sample_mask, sample_name = dataset_raw[indices[sample_idx]]
    
    print("\n[1/5] Grad-CAM heatmaps...")
    cam_ours = grad_cam_ours(ours_encoder, sample_image)
    cam_imagenet = grad_cam_imagenet(imagenet_encoder, sample_image_norm)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(sample_image.permute(1, 2, 0).numpy())
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(sample_mask.squeeze().numpy(), cmap='gray')
    axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    axes[2].imshow(sample_image.permute(1, 2, 0).numpy())
    axes[2].imshow(cam_ours, cmap='jet', alpha=0.5)
    axes[2].set_title(f'{OURS_NAME} Grad-CAM', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    axes[3].imshow(sample_image.permute(1, 2, 0).numpy())
    axes[3].imshow(cam_imagenet, cmap='jet', alpha=0.5)
    axes[3].set_title(f'{IMAGENET_NAME} Grad-CAM', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.tight_layout()
    plt.savefig('eval_results/gradcam_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved to: eval_results/gradcam_comparison.png")
    
    print("\n[2/5] Multi-layer feature activation maps (5 random samples)...")
    os.makedirs('eval_results/feature_maps', exist_ok=True)
    fm_seed = 2024
    fm_total = min(5, len(dataset_raw))
    fm_indices = np.random.RandomState(fm_seed).permutation(len(dataset_raw))[:fm_total]
    num_feature_samples = fm_total
    for i in range(num_feature_samples):
        idx = fm_indices[i]
        img, img_norm, mask, name = dataset_raw[idx]
        visualize_feature_maps_ours(ours_encoder, img, f'eval_results/feature_maps/sample{i+1:02d}_{OURS_NAME}.png')
        visualize_feature_maps_imagenet(imagenet_encoder, img_norm, f'eval_results/feature_maps/sample{i+1:02d}_{IMAGENET_NAME}.png')
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_feature_samples} groups")
    print(f"  Saved to: eval_results/feature_maps/ ({num_feature_samples} groups in total)")
    
    print("\n[3/5] t-SNE / UMAP feature space plots...")
    visualize_tsne_umap_both(features_ours, features_imagenet, labels_ours, 'eval_results/tsne_umap_comparison.png')
    print("  Saved to: eval_results/tsne_umap_comparison.png")
    
    print("\n[4/5] Confidence calibration plots...")
    calibration_curve_plot(probs_ours, y_test_ours, OURS_NAME, f'eval_results/calibration_{OURS_NAME}.png')
    print(f"  Saved to: eval_results/calibration_{OURS_NAME}.png")
    calibration_curve_plot(probs_img, y_test_img, IMAGENET_NAME, f'eval_results/calibration_{IMAGENET_NAME}.png')
    print(f"  Saved to: eval_results/calibration_{IMAGENET_NAME}.png")
    
    print("\n[5/5] Feature statistics...")
    print(f"\n{OURS_NAME} feature statistics:")
    print(f"  Mean: {features_ours.mean():.4f}")
    print(f"  Std: {features_ours.std():.4f}")
    
    print(f"\n{IMAGENET_NAME} feature statistics:")
    print(f"  Mean: {features_imagenet.mean():.4f}")
    print(f"  Std: {features_imagenet.std():.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(features_ours.flatten(), bins=50, alpha=0.7, color='#3498DB')
    axes[0].set_title(f'{OURS_NAME} Feature Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Count')
    axes[1].hist(features_imagenet.flatten(), bins=50, alpha=0.7, color='#E74C3C')
    axes[1].set_title(f'{IMAGENET_NAME} Feature Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Feature Value')
    axes[1].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('eval_results/feature_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved to: eval_results/feature_distribution.png")
    
    print("\n" + "=" * 60)
    print("Evaluation complete! Results saved in the eval_results/ directory")
    print("=" * 60)

    # ---- Summarize all test metrics into evaluate_result.txt in the project root ----
    summary = {
        'device': str(DEVICE),
        'image_dir': TRAIN_IMAGE_DIR,
        'mask_dir': TRAIN_MASK_DIR,
        'num_samples': total_samples,
        'ours_name': OURS_NAME, 'ours_weights': OURS_WEIGHTS,
        'ours_desc': OURS_DESC,
        'imagenet_name': IMAGENET_NAME, 'imagenet_weights': IMAGENET_WEIGHTS,
        'feature_shape_ours': tuple(int(x) for x in features_ours.shape),
        'feature_shape_imagenet': tuple(int(x) for x in features_imagenet.shape),
        'cluster_ours': {int(u): int(c) for u, c in
                         zip(*np.unique(labels_ours, return_counts=True))},
        'cluster_imagenet': {int(u): int(c) for u, c in
                             zip(*np.unique(labels_imagenet, return_counts=True))},
        'feature_stats_ours': (float(features_ours.mean()),
                               float(features_ours.std())),
        'feature_stats_imagenet': (float(features_imagenet.mean()),
                                   float(features_imagenet.std())),
        'results': results,
    }
    result_path = os.path.join(PROJECT_ROOT, 'evaluate_result.txt')
    write_result_txt(result_path, summary)
    print(f"  Test metric summary saved to: {result_path}")

    return results


if __name__ == '__main__':
    evaluate_all()

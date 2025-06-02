import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from model import initialize_weights, BiomarkerNet
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import os


def compute_metrics(y_true, y_pred):
    metrics = {}
    macro_f1_list = []
    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        macro_f1_list.append(f1)
        metrics[f'B{i+1}'] = {
            'f1': f1,
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
            'confusion_matrix': confusion_matrix(y_true[:, i], y_pred[:, i]).tolist()
        }
    metrics['macro_f1'] = np.mean(macro_f1_list)
    return metrics

def train_or_eval(model, dataloader, optimizer, criterion, device = 'cuda', train=True, reconstruction=False, recon_criterion=None, recon_weight=1.0):
    model.train() if train else model.eval()
    running_loss, total_valid = 0.0, 0
    all_labels, all_preds = [], []
    running_recon_loss = 0.0
    # print(f'I am running on device: {device}')
    with torch.set_grad_enabled(train):
        for image, label, clinical in dataloader:
            image = image.to(device)
            label = label.float().to(device)
            clinical = clinical.float().to(device)
            if train:
                optimizer.zero_grad()
            if model.mode == 'oct':
                out = model(image)
            else:
                out = model(image, clinical)
            if reconstruction:
                pred, recon_img = out
            else:
                pred = out
            mask = ~torch.isnan(label)
            labels_filled = torch.where(mask, label, torch.zeros_like(label))
            loss_mat = criterion(pred, labels_filled)
            masked_loss = (loss_mat * mask)
            loss = masked_loss.sum() / (mask.sum() + 1e-8)
            if reconstruction:
                recon_loss = recon_criterion(recon_img, image)
                loss += recon_weight * recon_loss
                running_recon_loss += recon_loss.item()
            running_loss += masked_loss.sum().item()
            total_valid += mask.sum().item()
            if train:
                loss.backward()
                optimizer.step()
            preds = (torch.sigmoid(pred) >= 0.5).float()
            all_labels.append(label.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    avg_loss = running_loss / (total_valid + 1e-8)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    metrics = compute_metrics(all_labels, all_preds)
    if reconstruction:
        metrics['recon_loss'] = running_recon_loss / len(dataloader)
    return avg_loss, metrics

def evaluate_model_on_test(model, test_loader, device, reconstruction=False, recon_criterion=None):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    running_recon_loss = 0.0
    with torch.no_grad():
        for image, label, clinical in test_loader:
            image = image.to(device)
            label = label.float().to(device)
            clinical = clinical.float().to(device)
            if model.mode == 'oct':
                out = model(image)
            else:
                out = model(image, clinical)
            if reconstruction:
                pred, recon_img = out
            else:
                pred = out
            probs = torch.sigmoid(pred)
            preds = (probs >= 0.5).float()
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    metrics = compute_metrics(all_labels, all_preds)
    if reconstruction:
        metrics['recon_loss'] = running_recon_loss / len(test_loader)
    return metrics, all_labels, all_preds, all_probs



def run_kfold_cv(full_dataset, test_loader, num_epochs, n_splits, batch_size, seed, mode, reconstruction, recon_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(full_dataset))
    acc = {f'fold_{i}': {'train': [], 'val': [], 'test': {}, 'best_macro_f1': 0, 'best_model': None} for i in range(n_splits)}
    acc_avg = {'train': [], 'val': [], 'test': [], 'test_std': []}
    all_macro_f1s = []
    all_best_models = []
    test_macro_f1s = []
    if reconstruction:
        recon_criterion = torch.nn.MSELoss()
    else:
        recon_criterion = None
    timestamp = time.localtime()
    # formatted_time = time.strftime("%Y%m%d_%H%M%S", timestamp)
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        
        formatted_time = time.strftime("%Y%m%d_%H%M%S", timestamp)
        writer = SummaryWriter(log_dir=f"runs/{formatted_time}/fold_{fold}_{mode}_{'recon' if reconstruction else 'norecon'}")
        # writer = SummaryWriter(log_dir=f"runs/fold_{fold}_{mode}_{'recon' if reconstruction else 'norecon'}")
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        model = BiomarkerNet(mode=mode, reconstruction=reconstruction).to(device)
        model.apply(initialize_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        best_macro_f1, best_model_state = 0, None
        best_epoch = 0
        for epoch in range(num_epochs):
            train_loss, train_metrics = train_or_eval(model, train_loader, optimizer, criterion, device, train=True,
                                                      reconstruction=reconstruction, recon_criterion=recon_criterion, recon_weight=recon_weight)
            val_loss, val_metrics = train_or_eval(model, val_loader, optimizer, criterion, device, train=False,
                                                  reconstruction=reconstruction, recon_criterion=recon_criterion, recon_weight=recon_weight)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MacroF1/train', train_metrics['macro_f1'], epoch)
            writer.add_scalar('MacroF1/val', val_metrics['macro_f1'], epoch)
            if reconstruction:
                writer.add_scalar('ReconLoss/train', train_metrics.get('recon_loss', 0), epoch)
                writer.add_scalar('ReconLoss/val', val_metrics.get('recon_loss', 0), epoch)
            for i in range(6):
                writer.add_scalar(f'F1/B{i+1}_train', train_metrics[f'B{i+1}']['f1'], epoch)
                writer.add_scalar(f'F1/B{i+1}_val', val_metrics[f'B{i+1}']['f1'], epoch)
            acc[f'fold_{fold}']['train'].append(train_metrics)
            acc[f'fold_{fold}']['val'].append(val_metrics)
            if val_metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = val_metrics['macro_f1']
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
        acc[f'fold_{fold}']['best_macro_f1'] = best_macro_f1
        acc[f'fold_{fold}']['best_model'] = best_model_state
        all_macro_f1s.append(best_macro_f1)
        all_best_models.append(best_model_state)
        torch.save(best_model_state, f"best_model_fold{fold}_{mode}_{'recon' if reconstruction else 'norecon'}.pth")
        model.load_state_dict(best_model_state)
        test_metrics = evaluate_model_on_test(model, test_loader, device, reconstruction=reconstruction, recon_criterion=recon_criterion)
        
        
        y_true_folds = []
        y_probs_folds = []
        test_metrics, y_true, y_pred, y_probs = evaluate_model_on_test(model, test_loader, device, reconstruction=reconstruction, recon_criterion=recon_criterion)
        
        # output_dir = f'plots/fold_{fold+1}'
        output_dir = f"runs/{formatted_time}/fold_{fold}_{mode}_{'recon' if reconstruction else 'norecon'}/plots/fold_{fold}"
        plot_roc_curves(y_true, y_probs, output_dir=output_dir, fold=fold, prefix='test')
        plot_conf_matrix(y_true, y_pred, output_dir=output_dir, fold=fold, prefix='test')
        
        y_true_folds.append(y_true)
        y_probs_folds.append(y_probs)
        all_y_true = np.concatenate(y_true_folds, axis=0)
        all_y_probs = np.concatenate(y_probs_folds, axis=0)
        # mean_roc_dir = 'plots/mean'
        mean_roc_dir = f"runs/{formatted_time}/fold_{fold}_{mode}_{'recon' if reconstruction else 'norecon'}/plots/mean"
        mean_aucs = plot_mean_roc_curve(all_y_true, all_y_probs, output_dir=mean_roc_dir, prefix='test')
        print(f"Mean ROC AUCs per biomarker (all folds): {[f'B{i}: {float(mean_aucs[i])}' for i in range(len(mean_aucs))]}")
        
        acc[f'fold_{fold}']['test'] = test_metrics
        test_macro_f1s.append(test_metrics['macro_f1'])
        writer.close()
    avg_macro_f1 = np.mean(all_macro_f1s)
    avg_macro_f1_test = np.mean(test_macro_f1s)
    std_macro_f1_test = np.std(test_macro_f1s)
    # Per-class
    test_f1s = [[] for _ in range(6)]
    for fold in range(n_splits):
        for i in range(6):
            test_f1s[i].append(acc[f'fold_{fold}']['test'][f'B{i+1}']['f1'])
    avg_test = {f'B{i+1}': {'f1_mean': np.mean(test_f1s[i]), 'f1_std': np.std(test_f1s[i])} for i in range(6)}
    acc_avg['test'] = avg_test
    acc_avg['macro_f1_test_mean'] = avg_macro_f1_test
    acc_avg['macro_f1_test_std'] = std_macro_f1_test
    return acc, acc_avg


def plot_roc_curves(y_true, y_probs, output_dir, fold=None, prefix='test'):
    os.makedirs(output_dir, exist_ok=True)
    n_classes = y_true.shape[1]
    aucs = []
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f'B{i+1} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves per Biomarker ({prefix} set{"" if fold is None else f", fold {fold+1}"})')
    plt.legend(loc="lower right")
    fname = os.path.join(output_dir, f'roc_{prefix}_{"fold" + str(fold+1) if fold is not None else "all"}.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    return aucs

def plot_conf_matrix(y_true, y_pred, output_dir, fold=None, prefix='test'):
    os.makedirs(output_dir, exist_ok=True)
    n_classes = y_true.shape[1]
    for i in range(n_classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix for B{i+1} ({prefix} set{"" if fold is None else f", fold {fold+1}"})')
        fname = os.path.join(output_dir, f'cm_B{i+1}_{prefix}_{"fold" + str(fold+1) if fold is not None else "all"}.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

def plot_mean_roc_curve(y_true, y_probs, output_dir, prefix='test'):
    os.makedirs(output_dir, exist_ok=True)
    n_classes = y_true.shape[1]
    plt.figure(figsize=(10, 7))
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label=f'B{i+1} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Mean ROC Curves per Biomarker ({prefix} set, all folds)')
    plt.legend(loc="lower right")
    fname = os.path.join(output_dir, f'roc_{prefix}_mean.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    return aucs

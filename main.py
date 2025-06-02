import argparse
from datasets import load_from_disk, load_dataset
from dataset import BiomarkerDataset, img_transforms, clinical_transforms
from train_eval import run_kfold_cv
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='fusion', choices=['oct', 'fusion'])
parser.add_argument('--reconstruction', action='store_true')
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--recon_weight', type=float, default=1.0)
parser.add_argument('--data_path', type=str, default="/home/jamshid/workspace/fundamental_exercise/removed_nans_filtered_olives_dataset")
parser.add_argument('--test_split', type=str, default="/home/jamshid/workspace/fundamental_exercise/olving_test_dataset")

args = parser.parse_args()

filtered_olives = load_from_disk(args.data_path)
biomarker_dataset = BiomarkerDataset(filtered_olives, img_transforms=img_transforms, clinical_transforms=clinical_transforms)
olives_test = load_from_disk(args.test_split)
# olives_test = load_dataset('gOLIVES/OLIVES_Dataset', 'biomarker_detection', split=args.test_split)
biomarker_dataset_test = BiomarkerDataset(olives_test, img_transforms=img_transforms, clinical_transforms=clinical_transforms)
test_loader = torch.utils.data.DataLoader(biomarker_dataset_test, batch_size=args.batch_size, shuffle=False)

acc, acc_avg = run_kfold_cv(
    biomarker_dataset,
    test_loader,
    num_epochs=args.num_epochs,
    n_splits=args.n_splits,
    batch_size=args.batch_size,
    seed=42,
    mode=args.mode,
    reconstruction=args.reconstruction,
    recon_weight=args.recon_weight
)

print("\nMean ± Std of Test Set F1 Across Folds:")
for b in range(6):
    print(f"B{b+1}: F1 = {acc_avg['test'][f'B{b+1}']['f1_mean']:.4f} ± {acc_avg['test'][f'B{b+1}']['f1_std']:.4f}")
print(f"Macro F1 (test): {acc_avg['macro_f1_test_mean']:.4f} ± {acc_avg['macro_f1_test_std']:.4f}")

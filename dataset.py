import torch
from torch.utils.data import Dataset
from torchvision import transforms

clinical_mean = torch.tensor([77.74157303, 336.96067416], dtype=torch.float32)
clinical_std = torch.tensor([9.63171814, 102.003825], dtype=torch.float32)

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2581297341430058], std=[0.17599633505555923])
])
clinical_transforms = transforms.Lambda(lambda x: (x - clinical_mean) / clinical_std)

class BiomarkerDataset(Dataset):
    def __init__(self, dataset, img_transforms=None, clinical_transforms=None):
        self.dataset = dataset
        self.img_transforms = img_transforms
        self.clinical_transforms = clinical_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[int(idx)]
        biomarkers = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
        label = torch.tensor([data[i] for i in biomarkers], dtype=torch.float32)
        bcva = data.get('BCVA', 0.0)
        cst = data.get('CST', 0.0)
        if bcva is None or (isinstance(bcva, float) and torch.isnan(torch.tensor(bcva))):
            bcva = 0.0
        if cst is None or (isinstance(cst, float) and torch.isnan(torch.tensor(cst))):
            cst = 0.0
        clinical = torch.tensor([bcva, cst], dtype=torch.float32)
        image = data['Image']
        if self.img_transforms:
            image = self.img_transforms(image)
            if image.shape[0] != 1:
                image = image[0, :, :][None, :, :]
        else:
            image = image.float()
        if self.clinical_transforms:
            clinical = self.clinical_transforms(clinical)
        return image, label, clinical

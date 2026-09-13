import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

CLASSES = ['normal', 'crack', 'scratch', 'dent', 'surface_defect']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

class ProductDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for cls_name in CLASSES:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, CLASS_TO_IDX[cls_name]))
                    
        print(f"[INFO] Loaded Dataset from {root_dir}: {len(self.samples)} total images across {len(CLASSES)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_image_transforms(image_size=(128, 128)):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

def create_image_dataloaders(raw_img_dir=os.path.join('data', 'raw', 'images'), batch_size=32, val_split=0.15, test_split=0.15, random_seed=42):
    train_transform, eval_transform = get_image_transforms()
    
    full_dataset = ProductDefectDataset(root_dir=raw_img_dir, transform=None)
    total_samples = len(full_dataset)
    
    if total_samples == 0:
        raise ValueError(f"No images found in {raw_img_dir}. Please run data/generate_image_data.py first.")
        
    val_size = int(total_samples * val_split)
    test_size = int(total_samples * test_split)
    train_size = total_samples - val_size - test_size
    
    generator = torch.Generator().manual_seed(random_seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    
    # Wrap subsets with proper transforms
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __len__(self):
            return len(self.subset)
            
        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

    train_transformed = TransformedSubset(train_ds, train_transform)
    val_transformed = TransformedSubset(val_ds, eval_transform)
    test_transformed = TransformedSubset(test_ds, eval_transform)
    
    train_loader = DataLoader(train_transformed, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_transformed, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_transformed, batch_size=batch_size, shuffle=False)
    
    print(f"[SUCCESS] DataLoaders created: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    create_image_dataloaders()

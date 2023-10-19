import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
import scipy.io
import argparse

class CustomDataset(Dataset):
    def __init__(self, img_folder, mat_folder, transform=None):
        self.img_folder = img_folder
        self.mat_folder = mat_folder
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.image_files[idx])
        image = read_image(img_path)
        mat_name = f"{idx+1}.mat"
        mat_path = os.path.join(self.mat_folder, mat_name)
        mat_data = scipy.io.loadmat(mat_path)
        target = torch.tensor(mat_data['target'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

def create_datasets(img_folder, mat_folder, train_size_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset(img_folder, mat_folder, transform)
    train_size = int(train_size_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def main(img_folder, mat_folder, train_size_ratio=0.8, batch_size=32):
    train_dataset, val_dataset = create_datasets(img_folder, mat_folder, train_size_ratio)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size)
    
    for images, targets in train_loader:
        # Your training code here
        pass

if __name__ == "__main__":
    # Configurable parameters
    IMG_FOLDER = "path_to_images_folder"
    MAT_FOLDER = "path_to_mat_folder"
    TRAIN_SIZE_RATIO = 0.8
    BATCH_SIZE = 32
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on custom data")
    parser.add_argument('--img_folder', type=str, required=True, default="/root/data/ieeg_nsd/shared1000",help='Path to the folder containing images')
    parser.add_argument('--mat_folder', type=str, required=True, default="/root/data/ieeg_nsd/wavelet_NSD1000",help='Path to the folder containing wavelet files')
    parser.add_argument('--train_size_ratio', type=float, default=0.8, help='Ratio of the dataset to be used for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    main(args.img_folder, args.mat_folder, args.train_size_ratio, args.batch_size)
    
    train_dataset, val_dataset = create_datasets(args.img_folder, args.mat_folder, args.train_size_ratio)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, args.batch_size)

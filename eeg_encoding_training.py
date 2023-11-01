import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
import torch.nn as nn
import torch.optim as optim
import scipy.io
import argparse
import glob
import re
import pickle
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from models.eeg_predictor import alexnet_layerwiseFWRF_eeg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, img_folder, mat_folder, transform=None):
        self.img_folder = img_folder
        self.mat_folder = mat_folder
        self.transform = transform
        # Load all target filenames
        self.target_files = sorted([f for f in os.listdir(mat_folder) if f.endswith('.mat')])
        
        # Create a dictionary of image filenames with the index as the key
        self.image_files_dict = {re.search(r"shared(\d{4})", f).group(1): f for f in os.listdir(img_folder) if f.endswith('.png')}
        
        #get size of wavelet spectrum
        sample_mat_name = f"shared01.mat"
        sample_mat_path = next(iter(glob.glob(os.path.join(self.mat_folder, sample_mat_name))), None)
        if sample_mat_path is None:
            raise FileNotFoundError(f"No mat file found for index 1")
        sample_mat_data = scipy.io.loadmat(sample_mat_path)
        array_key = next(iter(set(sample_mat_data.keys()) - {"__header__", "__version__", "__globals__"}))
        self.target_shape = sample_mat_data[array_key].shape
        
    def __len__(self):
        return len(self.target_files)
    
    def __getitem__(self, idx):
        # Extract the index from the target filename
        target_filename = self.target_files[idx]
        match = re.search(r"shared(\d{2,})", target_filename)
        if not match:
            raise ValueError(f"Unexpected filename format: {target_filename}")
        index = match.group(1).zfill(4)  # This is the extracted index

        # Check if corresponding image exists
        if index not in self.image_files_dict:
            raise ValueError(f"No image found for index {index}")
        img_path = os.path.join(self.img_folder, self.image_files_dict[index])

        # Construct the path for the target
        mat_path = os.path.join(self.mat_folder, target_filename)

        # Load the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load the target
        mat_data = scipy.io.loadmat(mat_path)
        array_key = next(iter(set(mat_data.keys()) - {"__header__", "__version__", "__globals__"}))
        target = torch.tensor(mat_data[array_key], dtype=torch.float32)

        return image, target

def create_datasets(img_folder, mat_folder, train_size_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CustomDataset(img_folder, mat_folder, transform)
    train_size = int(train_size_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    target_shape = dataset.target_shape
    
    return train_dataset, val_dataset, target_shape

def pearson_correlation_coefficient(pred, target):
    """
    Compute the Pearson Correlation Coefficient.
    """
    # Compute the means
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)

    # Compute the numerator and the denominators
    numerator = torch.sum((pred - pred_mean) * (target - target_mean))
    denominator = torch.sqrt(torch.sum((pred - pred_mean) ** 2) * torch.sum((target - target_mean) ** 2))

    # Avoid division by zero
    return numerator / (denominator + 1e-10)

def get_coordinates(i, width):
    x_axis = i // width
    y_axis = i % width
    return x_axis, y_axis

def main(img_folder, mat_folder, model_folder, writer_folder,train_size_ratio, batch_size, num_epochs, learning_rate):
    
    train_dataset, val_dataset, target_shape = create_datasets(img_folder, mat_folder, train_size_ratio)
    target_shape_size = target_shape[0]*target_shape[1]

    # Load all targets
    all_targets_train = []
    for _, target in train_dataset:
        all_targets_train.append(target)
    all_targets_train = torch.stack(all_targets_train).view(-1, target_shape_size)

    all_targets_val = []
    for _, target in val_dataset:
        all_targets_val.append(target)
    all_targets_val = torch.stack(all_targets_val).view(-1, target_shape_size)

    print(">All data loaded")

    # Initialize the dictionary to store the best correlation coefficient for each i with its coordinate
    best_corrs = {}

    # Train a model for each position
    for i in range(0, target_shape_size):
        # Set up TensorBoard writer
        if i%100 == 0:
            writer = SummaryWriter(os.path.join(writer_folder,f'{i}_model'))  # 'runs' is a common directory name for TensorBoard logs

        #Each pixel in spetrum as a target
        position_targets = all_targets_train[:, i]
        position_dataset = [(train_dataset[idx][0], position_targets[idx]) for idx in range(len(train_dataset))]
        train_loader_position = DataLoader(position_dataset, batch_size=batch_size, shuffle=True)

        position_targets_val = all_targets_val[:, i]
        position_dataset_val = [(val_dataset[idx][0], position_targets_val[idx]) for idx in range(len(val_dataset))]
        val_loader_position = DataLoader(position_dataset_val, batch_size=batch_size, shuffle=False)
        #print(">>>pixel data loaded")

        #Initialize training
        model = alexnet_layerwiseFWRF_eeg(device)
        model = model.to(device)
        print(">>>>>>Model loaded")
        # Dummy forward pass to initialize the readout layer
        dummy_input = torch.randn(1, 3, 227, 227).to(device)  # Assuming input shape is [batch_size, channels, height, width]
        _ = model(dummy_input)

        # Now initialize the optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        criterion = nn.MSELoss()
        best_corr_coeff = float('-inf')

        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            #training loop
            model.train()
            total_train_loss = 0
            correlations = []
            for batch_images, batch_labels in train_loader_position:
                #images = images.view(images.size(0), -1)
                # Move data to GPU if available
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                
                # Forward pass
                predictions = model(batch_images)
                loss = criterion(predictions, batch_labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                # Compute correlation for the current batch and store
                corr = pearson_correlation_coefficient(predictions.squeeze(), batch_labels.squeeze())
                correlations.append(corr.item())

            avg_train_loss = total_train_loss / len(train_loader_position)
            train_avg_corr = sum(correlations) / len(correlations)
            #print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Avg Correlation: {avg_corr:.4f}")

            if i%100 == 0:
                writer.add_scalar('Training Loss', avg_train_loss, epoch)
                writer.add_scalar('Avg training Correlation', train_avg_corr, epoch)
            
             # Validation Loop
            model.eval()
            total_val_loss = 0
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for batch_images, batch_labels in val_loader_position:
                    # Move data to GPU if available
                    batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                    
                    predictions = model(batch_images)
                    loss = criterion(predictions, batch_labels)
                    total_val_loss += loss.item()

                    # Compute correlation for the current batch and store
                    #corr = pearson_correlation_coefficient(predictions.squeeze(), batch_labels.squeeze())
                    #correlations.append(corr.item())
                    all_predictions.extend(predictions.squeeze().tolist())
                    all_labels.extend(batch_labels.squeeze().tolist())
            
            avg_val_loss = total_val_loss / len(val_loader_position)
            avg_corr = pearson_correlation_coefficient(torch.tensor(all_predictions), torch.tensor(all_labels))
            print(f"Position [{i + 1}/{target_shape_size}],Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Avg Correlation: {avg_corr:.4f}")
            
            if i%100 == 0:
                writer.add_scalar('Validation Loss', avg_val_loss, epoch)
                writer.add_scalar('Avg Validaton Correlation', avg_corr, epoch)
                writer.close
            if avg_corr > best_corr_coeff:
                best_corr_coeff = avg_corr
                torch.save(model.state_dict(), os.path.join(model_folder, f"{i}_best.pth"))
                #print("Saved best model with correlation coefficient:", best_corr_coeff)


        x_axis, y_axis = get_coordinates(i, target_shape[1])
        print(f"Position [{i + 1}/{target_shape_size}], Coordinates: ({x_axis},{y_axis}), Avg Training Correlation: {train_avg_corr:.4f}, Avg Validation Correlation: {avg_corr:.4f}")
        # Save the final model
        torch.save(model.state_dict(), os.path.join(model_folder, f"{i}_final.pth"))
        #print("Saved model from the final epoch.")

        #Save best_corrs with pixel coordinates
        best_corrs[i] = {
            "corr": best_corr_coeff,
            "coordinate": (x_axis, y_axis)  
        }
        if (i+1)%500==0:
            with open("./best_corrs_avg.pkl", "wb") as f:
                pickle.dump(best_corrs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default="/root/fdata/ieeg_nsd/shared1000",help='Path to the folder containing images')
    parser.add_argument('--mat_folder', type=str, default="/root/fdata/ieeg_nsd/wavelet_NSD1000_LTP3LTP4_avg",help='Path to the folder containing wavelet files')
    parser.add_argument('--model_folder', type=str, default="/root/fworkspace/ieeg_nsd/models")
    parser.add_argument('--writer_folder', type=str, default="./runs")
    parser.add_argument('--train_size_ratio', type=float, default=0.8, help='Ratio of the dataset to be used for training')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    main(args.img_folder, args.mat_folder, args.model_folder, args.writer_folder, args.train_size_ratio, args.batch_size, args.epoch, args.learning_rate)

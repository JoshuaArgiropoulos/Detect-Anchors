import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet18_Weights
from resNet import ResNet
import time  
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torch.multiprocessing import freeze_support
from formatData import FormatData  
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device, patience=6):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    early_stop_counter = 0
    total_start_time = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(average_epoch_loss)
        scheduler.step(average_epoch_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            val_loss = 0

            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

                val_loss += criterion(outputs, labels).item()


            accuracy = total_correct / total_samples
            avg_loss = val_loss / len(test_loader)
            val_losses.append(avg_loss)
    
            end_time = time.time()
            print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_epoch_loss:.4f}, '
                f'Accuracy: {accuracy:.4f},'
                f'Validation Loss: {avg_loss:.4f}, Time: {end_time - start_time:.2f} seconds')

            # Early stopping check
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f'Early stopping after {epoch + 1} epochs.')
                total_end_time = time.time()
                break

    total_end_time = time.time()
    print(f'Time: {((total_end_time - total_start_time) / 60.0):.2f} minutes')
    return train_losses, val_losses

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    val_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            val_loss += criterion(outputs, labels).item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = total_correct / total_samples
    avg_loss = val_loss / len(dataloader)

    return accuracy, avg_loss, all_labels, all_predictions

def main(args):
    print('Loading Data...')
    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using following device", device)
    
    dataset_root = 'data/Kitti8_ROIs'

    # Loading Train Dataset from FormatData
    train_dataset = FormatData(
        root_dir=dataset_root,
        mode='train',
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
        target_size=(150, 150)
    )

    # Loading Test Dataset from FormatData
    test_dataset = FormatData(
        root_dir=dataset_root,
        mode='test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),
        target_size=(150, 150)
    )

    # Train and test loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Finished Loading Data")
    
    # 'Car' and 'NoCar'
    num_classes = 2 
    
    # Initialize ResNet Model
    model = ResNet(num_classes, weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    
    
    # Hyperparameters
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')


    # Training Mode:
    if args.mode == "train":
        train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, args.num_epochs, device, patience=5)
        torch.save(model.state_dict(), args.output_path)
        
        # Loss Plot
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.legend()
        plt.savefig(args.loss_plot_path)
        plt.show()

    # Evaluation/Testing Mode:
    elif args.mode == "evaluate":
        model.load_state_dict(torch.load(args.pth_file))
        
        accuracy, avg_loss, all_labels, all_predictions = evaluate(model, test_loader, criterion, device)
        
        # Print out the accuracy and loss from testing
        print(f'Accuracy: {accuracy:.3f}, Loss: {avg_loss:.3f}')

        # Plotting the confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.show()

if __name__ == "__main__":
    # Argument Parse
    parser = argparse.ArgumentParser(description="YODA DETECTION!")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch Size")
    parser.add_argument("--pth_file", type=str, default="model.pth", help="Path to save the trained model")
    parser.add_argument("--loss_plot_path", type=str, default="Loss_Plot", help="Path for Loss Plot")
    parser.add_argument("--num_epochs", type=int, default=40, help="# of Epochs")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train", help="Train or Evaluate")
    parser.add_argument("--output_path", type=str, default="./model.pth", help="Output path")
    args = parser.parse_args()

    main(args)
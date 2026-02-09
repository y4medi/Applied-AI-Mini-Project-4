import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
import numpy as np

# Import our modules
from model import FashionClassifier
from utils import get_data_loaders, plot_training_curves, plot_confusion_matrix_heatmap, get_device

def train(model, loader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm for a progress bar
    pbar = tqdm(loader, desc="Training", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix(loss=running_loss/len(loader), acc=100.*correct/total)

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    """Evaluates the model on the test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_labels, all_preds

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a FashionMNIST Classifier")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # New arguments for experiments
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[512], help='List of hidden layer sizes')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu', 'elu', 'tanh'], help='Activation function')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--batch_norm', action='store_true', help='Use Batch Normalization in classifier')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'adamw'], help='Optimizer')
    
    args = parser.parse_args()

    # Setup
    device = get_device()
    print(f"Using device: {device}")
    print(f"Configuration: {args}")

    # Data
    train_loader, test_loader, classes = get_data_loaders(batch_size=args.batch_size)
    print(f"Classes: {classes}")

    # Model
    model = FashionClassifier(
        num_classes=10,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        dropout_rate=args.dropout,
        use_bn=args.batch_norm
    ).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Training Loop
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_acc = 0.0

    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        test_loss, test_acc, all_labels, all_preds = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), './results/best_model.pth')
            print("Best model saved!")

    # Plotting
    print("\nGenerating plots...")
    plot_training_curves(train_losses, test_losses, train_accs, test_accs)
    
    # Final Evaluation for Confusion Matrix
    print("Generating confusion matrix...")
    _, _, final_labels, final_preds = evaluate(model, test_loader, criterion, device)
    plot_confusion_matrix_heatmap(final_labels, final_preds, classes)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
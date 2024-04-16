import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MNIST
from dataset_regularization import MNIST_2
from model import LeNet5, CustomMLP
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, trn_loader, device, criterion, optimizer):
    model.train()
    trn_loss = 0
    correct = 0
    for data, target in trn_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    trn_loss /= len(trn_loader.dataset)
    acc = 100. * correct / len(trn_loader.dataset)
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    model.eval()
    tst_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tst_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            tst_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    tst_loss /= len(tst_loader.dataset)
    acc = 100. * correct / len(tst_loader.dataset)
    return tst_loss, acc

def main():
    batch_size = 64
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # LeNet5
    trn_dataset = MNIST('../deep_hw2/mnist-classification/data/train/')
    tst_dataset = MNIST('../deep_hw2/mnist-classification/data/test/')
    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=batch_size, shuffle=False)

    lenet5 = LeNet5().to(device)
    cmlp = CustomMLP().to(device)
    print(f'LeNet-5 parameter count: {count_parameters(lenet5)}')
    print(f'Custom MLP parameter count: {count_parameters(cmlp)}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lenet5.parameters(), lr=0.01, momentum=0.9)

    trn_losses, trn_accs = [], []
    tst_losses, tst_accs = [], []

    for epoch in range(epochs):
        trn_loss, trn_acc = train(lenet5, trn_loader, device, criterion, optimizer)
        tst_loss, tst_acc = test(lenet5, tst_loader, device, criterion)
        trn_losses.append(trn_loss)
        trn_accs.append(trn_acc)
        tst_losses.append(tst_loss)
        tst_accs.append(tst_acc)
        print(f'Epoch {epoch+1}/{epochs}, TrainLoss: {trn_loss:.4f}, TrainAcc: {trn_acc:.2f}%, TestLoss: {tst_loss:.4f}, TestAcc: {tst_acc:.2f}%')

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(trn_losses, label='Train Loss')
    plt.plot(tst_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(trn_accs, label='Train Accuracy')
    plt.plot(tst_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./lenet5_curves.png')

    # CustomMLP
    optimizer = optim.SGD(cmlp.parameters(), lr=0.01, momentum=0.9)
    trn_losses, trn_accs = [], []
    tst_losses, tst_accs = [], []

    for epoch in range(epochs):
        trn_loss, trn_acc = train(cmlp, trn_loader, device, criterion, optimizer)
        tst_loss, tst_acc = test(cmlp, tst_loader, device, criterion)
        trn_losses.append(trn_loss)
        trn_accs.append(trn_acc)
        tst_losses.append(tst_loss)
        tst_accs.append(tst_acc)
        print(f'Epoch {epoch+1}/{epochs}, TrainLoss: {trn_loss:.4f}, TrainAcc: {trn_acc:.2f}%, TestLoss: {tst_loss:.4f}, TestAcc: {tst_acc:.2f}%')

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(trn_losses, label='Train Loss')
    plt.plot(tst_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(trn_accs, label='Train Accuracy')
    plt.plot(tst_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./cmlp_curves.png')

    # LeNet5 - Regularization
    trn_dataset = MNIST_2('../deep_hw2/mnist-classification/data/train/')
    tst_dataset = MNIST_2('../deep_hw2/mnist-classification/data/test/')
    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=batch_size, shuffle=False)

    lenet5 = LeNet5().to(device)
    print(f'LeNet-5 parameter count: {count_parameters(lenet5)}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lenet5.parameters(), lr=0.01, momentum=0.9)

    trn_losses, trn_accs = [], []
    tst_losses, tst_accs = [], []

    for epoch in range(epochs):
        trn_loss, trn_acc = train(lenet5, trn_loader, device, criterion, optimizer)
        tst_loss, tst_acc = test(lenet5, tst_loader, device, criterion)
        trn_losses.append(trn_loss)
        trn_accs.append(trn_acc)
        tst_losses.append(tst_loss)
        tst_accs.append(tst_acc)
        print(f'Epoch {epoch+1}/{epochs}, TrainLoss: {trn_loss:.4f}, TrainAcc: {trn_acc:.2f}%, TestLoss: {tst_loss:.4f}, TestAcc: {tst_acc:.2f}%')

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(trn_losses, label='Train Loss')
    plt.plot(tst_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(trn_accs, label='Train Accuracy')
    plt.plot(tst_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./lenet5_curves_regularization.png')

if __name__ == '__main__':
    main()

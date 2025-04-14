import torch 
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, criterion, optimizer, schedular, num_epoche, device):
    best_acc= 0.0
    for epoche in range(num_epoche):
        print(f'Epoche {epoche+1}/{num_epoche}')
        print('-' * 10)

        #Training Phase 
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels  in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backword()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if schedular:
            schedular.step()
        epoche_loss = running_loss / len(train_loader.dataset)
        epoche_accuracy = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoche_loss:.4f} Acc: {epoche_loss:.4f}')

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        if epoche_accuracy > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved new best model with accuracy {best_acc:.4f}')
        return model

def evaluate_model(model, test_loader, device):
    model.eval()
    running_corrects = 0 
    with torch.no_grad():
        for inputs , labels in test_loader:
            inputs , labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {acc:.4f}')
    return acc


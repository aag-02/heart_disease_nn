import torch
from torch import nn
from src.model import TheNet
from src.evaluate import evaluate
import matplotlib.pyplot as plt

def train_model(train_loader, test_loader, device, num_epochs=100, learning_rate=1e-4):
    model = TheNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        avg_train_loss = epoch_loss / total
        train_accuracy = 100 * correct / total
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)

        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_accuracy)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] | '
                  f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | '
                  f'Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%')

    return model


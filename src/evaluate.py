import torch
from sklearn.metrics import classification_report, roc_auc_score

def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            epoch_loss += loss.item() * X_batch.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    avg_loss = epoch_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def final_evaluation(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            scores = torch.sigmoid(outputs)
            predictions = (scores > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions))
    print(f'ROC AUC Score: {roc_auc_score(all_labels, all_scores):.2f}')


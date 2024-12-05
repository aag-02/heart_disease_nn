import torch
from src.utils import set_seed
from src.data_preprocessing import load_and_preprocess_data
from src.train import train_model
from src.evaluate import final_evaluation

def main():
    # Reproducibility
    set_seed()
    
    # Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    file_path = 'data/raw/heart_disease_uci.csv'
    train_loader, test_loader = load_and_preprocess_data(file_path)
    model = train_model(train_loader, test_loader, device)
    
    # Final Evaluation
    from src.evaluate import final_evaluation
    final_evaluation(model, test_loader, device)

if __name__ == '__main__':
    main()


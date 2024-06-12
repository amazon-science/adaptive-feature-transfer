import torch
from torch import nn, optim
from data import get_dataset, split_train
import numpy as np
import os
import fire
from copy import deepcopy
from tqdm import tqdm

def train_autoencoder(encoder, decoder, train_features, test_features, epochs=500, batch_size=1024, lr=3e-4, early_stop_patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()

    best_test_loss = float('inf')
    best_encoder_state = None
    best_decoder_state = None
    patience = early_stop_patience

    for epoch in (pbar := tqdm(range(epochs))):
        # Training loop
        encoder.train()
        decoder.train()
        train_loss = 0
        for i in range(0, len(train_features), batch_size):
            batch = train_features[i:i+batch_size].to(device)
            optimizer.zero_grad()
            encoded = encoder(batch)
            decoded = decoder(encoded)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_features)

        # Validation loop
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            test_loss = 0
            for i in range(0, len(test_features), batch_size):
                batch = test_features[i:i+batch_size].to(device)
                encoded = encoder(batch)
                decoded = decoder(encoded)
                loss = criterion(decoded, batch)
                test_loss += loss.item() * len(batch)
            test_loss /= len(test_features)

        # tqdm.write(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        pbar.set_description(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_encoder_state = deepcopy(encoder.state_dict())
            best_decoder_state = deepcopy(decoder.state_dict())
            patience = early_stop_patience
        else:
            patience -= 1
            if patience == 0:
                tqdm.write(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model weights
    encoder.load_state_dict(best_encoder_state)
    decoder.load_state_dict(best_decoder_state)

    return encoder, decoder

def run_logme(model_class, dataset, train_frac):
    feat_path = f'./features/{model_class}_{dataset}.pt'
    assert os.path.exists(feat_path), f'feature file {feat_path} not found'
    features = torch.load(feat_path)
    train_features = features['train']
    test_features = features['test']
    # print feature shapes
    print(f'train_features: {train_features.shape}')
    print(f'test_features: {test_features.shape}')
    # train autoencoder
    d = train_features.shape[1]
    encoder = nn.Sequential(
        nn.Linear(d, d),
        nn.ReLU(),
        nn.Linear(d, d//2),
        nn.ReLU(),
        nn.Linear(d//2, d//2),
        nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.Linear(d//2, d//2),
        nn.ReLU(),
        nn.Linear(d//2, d),
        nn.ReLU(),
        nn.Linear(d, d),
    )
    # train MLP with L2 reconstruction, early stop on test loss
    encoder, decoder = train_autoencoder(encoder, decoder, train_features, test_features)
    
    ae_features = deepcopy(features)
    ae_features['train'] = encoder(train_features.cuda()).detach().cpu()
    ae_features['test'] = encoder(test_features.cuda()).detach().cpu()
    torch.save(ae_features, f'./features/{model_class}_{dataset}_ae.pt')
    print(f'Autoencoder features saved to ./features/{model_class}_{dataset}_ae.pt')

def main(model_class, dataset, train_frac=1):
    run_logme(model_class, dataset, train_frac)

if __name__ == '__main__':
    fire.Fire(main)
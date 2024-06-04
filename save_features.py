import fire
import os
import torch
from data import get_dataset, get_loader
import models
import utils as u
from functools import partial
from tqdm import tqdm

def save_features(train_loader, test_loader, model, feature_path, debug):
    # if exactly 1 GPU is available
    if torch.cuda.device_count() == 1:
        model.cuda()
    else:
        print('Multiple GPUs detected, not moving model to GPU explicitly')
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(train_loader):
            if isinstance(batch, tuple) or isinstance(batch, list):
                inputs = batch[0]
            else:
                inputs = batch
                if 'labels' in inputs:
                    inputs.pop('labels') # remove for hf datasets
            if hasattr(inputs, 'items'):
                inputs = {k: v.cuda() for k, v in inputs.items()}
            else:
                inputs = inputs.cuda()
            feat = model(inputs).detach().cpu()
            features.append(feat)
            if debug: 
                break
        features = torch.cat(features, dim=0)

    test_features = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if isinstance(batch, tuple) or isinstance(batch, list):
                inputs = batch[0]
            else:
                inputs = batch
                if 'labels' in inputs:
                    inputs.pop('labels') # remove for hf datasets
            if hasattr(inputs, 'items'):
                inputs = {k: v.cuda() for k, v in inputs.items()}
            else:
                inputs = inputs.cuda()
            feat = model(inputs).detach().cpu()
            # test_features.append(feat)
            if debug: 
                break
        exit()
        test_features = torch.cat(test_features, dim=0)
        
    print(f'Features: {features.size()}', f'Test features: {test_features.size()}')
    feature_dict = {'train': features, 'test': test_features}
    if not debug:
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        torch.save(feature_dict, feature_path)
        print(f'Saved features to {feature_path}')

def main(model_class, dataset, batch_size=128, num_workers=0, save_path=None, debug=False, **kwargs):
    assert save_path is not None, "Please specify a save_path"
    args = locals()
    u.pretty_print_dict(args)
    # out_dim=0 should cause the model to return the features
    model, get_transform, tokenizer, input_collate_fn = models.create_model(model_class, out_dim=0, pretrained=True, extract_features=True, **kwargs)
    model.eval()
    train_test_dataset = get_dataset(dataset, get_transform, tokenizer, no_augment=True) # (train, test)
    train_test_loaders = [get_loader(ds, batch_size, num_workers=num_workers, shuffle=False, input_collate_fn=input_collate_fn) for ds in train_test_dataset]
    train_loader, test_loader = train_test_loaders
    save_features(train_loader, test_loader, model, save_path, debug)
    
if __name__ == '__main__':
    fire.Fire(main)
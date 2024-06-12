import fire
import os
import torch
import wandb
import numpy as np

from train import train
from data import get_dataset, get_loader, split_train, get_out_dim
import models
from models import Concat, LinearModel, ProductLinearModel
from prior import UniformPrior, get_prior, get_btune_prior
import utils as u
from data import FeatureDataset

def main(model_class, dataset, train_frac, use_val, method, steps, eval_steps=1000, prec=0, pretrained_models=None, init_model=None, batch_size=128, optimizer='sgd', lr=1e-3, wd=0, learn_scales=False, prior_lr=1e-2, prior_freq=1, prior_pretrain_steps=0, ckpt_dir=None, seed=0, use_wandb=False, run_name=None, no_augment=True, cache=False, **kwargs):
    assert no_augment or method == 'init', 'Must not use augmentation unless method == init'
    u.set_seed(seed)
    if dataset in ['mnli', 'qqp', 'qnli']:
        steps *= 4
    args = locals()
    u.pretty_print_dict(args)
    assert method in ['base', 'init', 'lconcat', 'fconcat', 'concat', 'aft', 'kd', 'ft', 'rkd', 'kkd', 'aft_rbf', 'aft_no_kernel', 'aft_dense', 'btune'], f'Unknown method {method}'
    out_dim = get_out_dim(dataset)
    # randomly init model
    model, get_transform, tokenizer, input_collate_fn = models.create_model(model_class, out_dim=out_dim, pretrained=False, **kwargs) # call get_transform with train=True/False to get train/test transforms
    # train val test split
    train_ds, test_ds = get_dataset(dataset, get_transform, tokenizer, no_augment, cache)
    val_frac = 0.1 if use_val else 0
    train_ds, val_ds = split_train(train_ds, train_frac, val_frac)
    train_indices = train_ds.indices
    val_indices = val_ds.indices
    print(f'Training set size: {len(train_ds)}')
    print(f'Validation set size: {len(val_ds)}')
    raw_train_ds, raw_test_ds = get_dataset(dataset, get_transform, tokenizer, no_augment=no_augment, cache=cache)
    # pretrained models
    if pretrained_models not in [None, 'none']:
        pretrained_models = pretrained_models.split(',')
        if not isinstance(pretrained_models, list):
            pretrained_models = (pretrained_models,)
        pretrained_models = [model.strip() for model in pretrained_models]
        print(f'Pretrained models: {pretrained_models}')
        if method == 'ft':
            feature_paths = [f'./features/{model}_{dataset}_ae.pt' for model in pretrained_models]
        else:
            feature_paths = [f'./features/{model}_{dataset}.pt' for model in pretrained_models]
        for path in feature_paths:
            assert os.path.exists(path), f'Feature path {path} does not exist'
        # annotate dataset with features
        train_ds = FeatureDataset(raw_train_ds, feature_paths, indices=train_indices, split='train') # (x, f, y)
        val_ds = FeatureDataset(raw_train_ds, feature_paths, indices=val_indices, split='train') # (x, f, y)
        test_ds = FeatureDataset(raw_test_ds, feature_paths, indices=None, split='test') # (x, f, y)
    else:
        # dummy feature dataset
        prec = 0
        train_ds = FeatureDataset(raw_train_ds, None, indices=train_indices, split='train') # (x, f, y)
        val_ds = FeatureDataset(raw_train_ds, None, indices=val_indices, split='train') # (x, f, y)
        test_ds = FeatureDataset(raw_test_ds, None, indices=None, split='test') # (x, f, y)
    # create loaders
    # 0 workser if torchvision
    num_workers = 0 #if dataset in ['cifar10', 'cifar100', 'flowers', 'pets'] else 4
    if use_val:
        print('---------- Evaluating on val set ----------')
        loaders = [get_loader(ds, batch_size, num_workers=num_workers, shuffle=(ds is train_ds), input_collate_fn=input_collate_fn) for ds in [train_ds, val_ds]]
    else:
        loaders = [get_loader(ds, batch_size, num_workers=num_workers, shuffle=(ds is train_ds), input_collate_fn=input_collate_fn) for ds in [train_ds, test_ds]]
    

    if method in ['init', 'aft', 'aft_no_kernel', 'aft_rbf', 'aft_dense', 'kd', 'ft', 'rkd', 'kkd', 'btune'] and init_model is not None:
        print(f'Init model: {init_model}')
        init_model = models.create_model(init_model, out_dim=out_dim, pretrained=True, **kwargs)[0]
    else:
        init_model = None

    if method == 'fconcat':
        train_ds = FeatureDataset(raw_train_ds, feature_paths, indices=train_indices, split='train', feature_only=True) # (f, y)
        test_ds = FeatureDataset(raw_test_ds, feature_paths, indices=None, split='test', feature_only=True) # (f, y)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        loaders = [train_loader, test_loader]
        if dataset == 'snli-ve':
            model = ProductLinearModel(train_ds.num_features, out_dim)    
        else:
            model = LinearModel(train_ds.num_features, out_dim)
    elif method == 'concat':
        pretrained_models = [models.create_model(model, out_dim=0, pretrained=True, **kwargs)[0] for model in pretrained_models]
        x = next(iter(loaders[0]))[0]
        num_features = sum([m(x).shape[-1] for m in pretrained_models])
        model = Concat(pretrained_models, num_features=num_features, out_dim=out_dim)

    if method == 'aft':
        prior = get_prior(model.feat_dim, train_ds, prec, learn_scales, tensor_product=(dataset=='snli-ve'), prior_type='kernel')
    elif method == 'aft_dense':
        prior = get_prior(model.feat_dim, train_ds, prec, learn_scales, tensor_product=(dataset=='snli-ve'), prior_type='kernel_dense')
    elif method == 'kkd':
        prior = get_prior(model.feat_dim, train_ds, prec, learn_scales, tensor_product=(dataset=='snli-ve'), prior_type='kkd')
    elif method == 'aft_rbf':
        prior = get_prior(model.feat_dim, train_ds, prec, learn_scales, tensor_product=(dataset=='snli-ve'), prior_type='kernel_rbf')
    elif method == 'aft_no_kernel':
        prior = get_prior(model.feat_dim, train_ds, prec, learn_scales, tensor_product=(dataset=='snli-ve'), prior_type='feature')
    elif method in ['kd', 'rkd', 'ft']:
        prior_lr = lr
        prior = get_prior(model.feat_dim, train_ds, prec, learn_scales, tensor_product=(dataset=='snli-ve'), prior_type=method)
    elif method == 'btune':
        prior = get_btune_prior(model_class, dataset, train_ds, prec, train_frac)
    else:
        prior = UniformPrior()
    
    if no_augment:
        loaders = [[d for d in loader] if loader and len(loader) <= 100 else loader for loader in loaders]

    hypers = {'optimizer': optimizer, 'lr': lr, 'steps': steps, 'eval_steps': eval_steps, 'wd': wd, 'prior_lr': prior_lr, 'prior_pretrain_steps': prior_pretrain_steps, 'prior_freq': prior_freq}
    if use_wandb:
        with wandb.init(project='transfer', config=args, save_code=True, name=run_name) as wandb_run:
            train(loaders, model, init_model, prior, wandb_run, hypers)
    else:
        train(loaders, model, init_model, prior, None, hypers)

    # save model
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), f'{ckpt_dir}/model.pt')
        # save prior
        if hasattr(prior, 'state_dict'):
            torch.save(prior.state_dict(), f'{ckpt_dir}/prior.pt')

if __name__ == '__main__':
    fire.Fire(main)
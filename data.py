import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision import transforms, datasets
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_paths, indices, split='train', feature_only=False):
        self.feature_only = feature_only
        self.feature_paths = feature_paths
        self.dataset = dataset
        if indices is None:
            indices = list(range(len(dataset)))
        self.indices = indices
        feats = []
        if feature_paths is None:
            print('Using dummy features')
            self.features = torch.zeros(len(self.dataset), 1) # (n, 1) dummy features
        else:
            for feature_path in feature_paths:
                assert os.path.exists(feature_path), f'Feature path {feature_path} does not exist'
            for feature_path in feature_paths:
                feat = torch.load(feature_path)[split]
                assert len(self.dataset) == len(feat), f'Feature path {feature_path} has {len(feat)} entries but dataset has {len(self.dataset)}'
                feats.append(feat[indices])
            self.features = torch.cat(feats, dim=1) # (n, d)
        self.feat_dims = [feat.size(1) for feat in feats]
        self.num_features = sum(self.feat_dims)
        print(f'Feature dims: {self.feat_dims}')
        print(f'Feature dataset: {self.features.size()}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f = self.features[idx]
        d = self.dataset[self.indices[idx]]
        if isinstance(d, tuple):
            x, y = d
        else:
            y = d.pop('label')
            x = d
        if self.feature_only:
            return f, y
        else:
            return x, f, y
        

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = self.preload_dataset()
        if isinstance(dataset, torch.utils.data.Subset):
            self.indices = dataset.indices

    def preload_dataset(self):
        print('Preloading dataset...')
        cache = [None]*len(self.dataset)
        for idx in tqdm(range(len(self.dataset))):
            cache[idx] = self.dataset[idx]
        return cache

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.dataset)

def get_cifar_transform(train):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

default_get_transform = {
    'cifar10': get_cifar_transform,
    'cifar100': get_cifar_transform,
}

def get_dataset(dataset, get_transform=None, tokenizer=None, no_augment=True, cache=False):
    if dataset == 'none':
        return None, None, None
    dataset = dataset.lower()
    # assert dataset in ['cifar10', 'cifar100'], f'Unknown dataset {dataset}'
    assert dataset in ['cifar10', 'cifar100', 'flowers', 'pets', 'aircraft', 'dtd', 'food',
                       'imdb', 'boolq', 'snli-ve', 'snli-ve-img', 'snli-ve-txt', 'cola', 'rte', 'mrpc', 'mnli', 'sst2', 'wnli', 'qnli', 'qqp'], f'Unknown dataset {dataset}'
    if not get_transform:
        print(f'Using default transform for {dataset}')
        get_transform = default_get_transform[dataset] if dataset in default_get_transform else lambda train: lambda x: x
    train_transform = get_transform(train=not no_augment) # no_augment overrides train transform, used for prior
    test_transform = get_transform(train=False)
    print('Train transform:')
    print(train_transform)
    print('Test transform:')
    print(test_transform)
    if dataset == 'snli-ve':
        train_ds = load_dataset('Multimodal-Fatima/SNLI-VE_train', ignore_verifications=True)['train']
        test_ds = load_dataset('Multimodal-Fatima/SNLI-VE_test', ignore_verifications=True)['test']
        if train_transform is not None and test_transform is not None:
            train_ds.set_transform(train_transform)
            test_ds.set_transform(test_transform)
        else:
            # linear prob eval
            print('Assuming linear prob eval')
            # drop all but label columns
            cols = train_ds.column_names
            to_remove = [c for c in cols if c not in ['label']]
            train_ds = train_ds.remove_columns(to_remove)
            test_ds = test_ds.remove_columns(to_remove)
    elif dataset == 'snli-ve-img':
        # image only
        # train_transform: img -> img
        train_ds = load_dataset('Multimodal-Fatima/SNLI-VE_train', ignore_verifications=True)['train']
        test_ds = load_dataset('Multimodal-Fatima/SNLI-VE_test', ignore_verifications=True)['test']
        train_ds.set_transform(lambda x: {'image': [train_transform(img) for img in x['image']]})
        test_ds.set_transform(lambda x: {'image': [test_transform(img) for img in x['image']]})
    elif dataset == 'snli-ve-txt':
        # text only
        assert train_transform == test_transform == None, 'train_transform and test_transform must be None for text only'
        assert tokenizer is not None, 'Must provide tokenizer'
        train_ds = load_dataset('Multimodal-Fatima/SNLI-VE_train', ignore_verifications=True)['train']
        test_ds = load_dataset('Multimodal-Fatima/SNLI-VE_test', ignore_verifications=True)['test']
        train_ds.set_transform(lambda x: tokenizer(x["hypothesis"], truncation=True))
        test_ds.set_transform(lambda x: tokenizer(x["hypothesis"], truncation=True))
    elif dataset == 'cifar10':
        train_ds = datasets.CIFAR10(root='~/data', train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR10(root='~/data', train=False, download=True, transform=test_transform)
    elif dataset == 'cifar100':
        train_ds = datasets.CIFAR100(root='~/data', train=True, download=True, transform=train_transform)
        test_ds = datasets.CIFAR100(root='~/data', train=False, download=True, transform=test_transform)
    elif dataset == 'flowers':
        train_ds = datasets.Flowers102(root='~/data', split='train', download=True, transform=train_transform)
        test_ds = datasets.Flowers102(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'pets':
        train_ds = datasets.OxfordIIITPet(root='~/data', split='trainval', download=True, transform=train_transform)
        test_ds = datasets.OxfordIIITPet(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'aircraft':
        train_ds = datasets.FGVCAircraft(root='~/data', split='trainval', download=True, transform=train_transform)
        test_ds = datasets.FGVCAircraft(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'dtd':
        train_ds = datasets.DTD(root='~/data', split='train', download=True, transform=train_transform)
        test_ds = datasets.DTD(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'food':
        train_ds = datasets.Food101(root='~/data', split='train', download=True, transform=train_transform)
        test_ds = datasets.Food101(root='~/data', split='test', download=True, transform=test_transform)
    elif dataset == 'imdb':
        imdb = load_dataset("imdb")
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = imdb['train']
            test_ds = imdb['test']
        else:
            tokenizer.truncation_side = 'left'
            postfix = 'Overall, the sentiment of my review is'
            preprocess_function = lambda x: tokenizer([t + postfix for t in x['text']], truncation=True)
            tokenized_imdb = imdb.map(preprocess_function, batched=True)
            tokenized_imdb = tokenized_imdb.remove_columns(["text"])
            train_ds = tokenized_imdb['train']
            test_ds = tokenized_imdb['test']
    elif dataset == 'boolq':
        dataset = load_dataset('super_glue', 'boolq')
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Question: {q}\nReference: {p}\nAnswer: ' for q, p in zip(x['question'], x['passage'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["question", "passage", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'mnli':
        dataset = load_dataset('glue', 'mnli') # (premise, hypothesis, label)
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation_matched']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Premise: {p}\nHypothesis: {h}\nDoes the premise entail the hypothesis? Answer: ' for p, h in zip(x['premise'], x['hypothesis'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["premise", "hypothesis", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation_matched']
    elif dataset == 'mrpc':
        dataset = load_dataset('glue', 'mrpc') # (sentence1, sentence2, label (equivalent or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Sentence 1: {s1}\nSentence 2: {s2}\nIs Sentence 1 equivalent to Sentence 2? Answer: ' for s1, s2 in zip(x['sentence1'], x['sentence2'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'sst2':
        dataset = load_dataset('glue', 'sst2') # (sentence, label (positive or negative))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation'] 
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Review: "{s}"\nSentiment: ' for s in x['sentence']], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'cola':
        dataset = load_dataset('glue', 'cola') # (sentence, label (sentence is grammatical or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Is the sentence "{s}" grammatical? Answer: ' for s in x['sentence']], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'qnli':
        dataset = load_dataset('glue', 'qnli') # (question, sentence, label (entailment or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'right'
            preprocess_function = lambda x: tokenizer([f'Question: {q}\nSentence: {s}\nDoes the sentence answer the question? Answer: ' for q, s in zip(x['question'], x['sentence'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["question", "sentence", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'rte':
        dataset = load_dataset('glue', 'rte') # (sentence1, sentence2, label (entailment or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Sentence 1: {s1}\nSentence 2: {s2}\nDoes Sentence 1 entail Sentence 2? Answer: ' for s1, s2 in zip(x['sentence1'], x['sentence2'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    elif dataset == 'qqp':
        dataset = load_dataset('glue', 'qqp') # (question1, question2, label (equivalent or not))
        if tokenizer is None:
            print('No tokenizer provided, dataset will be untokenized')
            train_ds = dataset['train']
            test_ds = dataset['validation']
        else:
            tokenizer.truncation_side = 'left'
            preprocess_function = lambda x: tokenizer([f'Question 1: {q1}\nQuestion 2: {q2}\nAre Question 1 and Question 2 equivalent? Answer: ' for q1, q2 in zip(x['question1'], x['question2'])], truncation=True)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns(["question1", "question2", "idx"])
            train_ds = tokenized_dataset['train']
            test_ds = tokenized_dataset['validation']
    return train_ds, test_ds

def split_train(train_ds, train_frac, val_frac):
    train_frac = min(train_frac, 1 - val_frac) # for explicitly subsampling train set
    train_ds, val_ds, _ = random_split(train_ds, [train_frac, val_frac, 1 - (train_frac + val_frac)], generator=torch.Generator().manual_seed(42))
    return train_ds, val_ds

def get_loader(ds, batch_size, num_workers=0, shuffle=False, input_collate_fn=None):
    if input_collate_fn is not None:
        if isinstance(ds, FeatureDataset):
            if not ds.feature_only:
                # (x, f, y) process x with input_collate_fn
                collate_fn = lambda batch: (input_collate_fn([b[0] for b in batch]), torch.stack([b[1] for b in batch]), torch.tensor([b[2] for b in batch]))
            else:
                collate_fn = None
        else:
            d = ds[0]
            if isinstance(d, tuple):
                # (x, y) process x with input_collate_fn
                collate_fn = lambda batch: (input_collate_fn([b[0] for b in batch]), torch.tensor([b[1] for b in batch]))
            else:
                # x process x with input_collate_fn
                collate_fn = lambda batch: input_collate_fn(batch)
    else:
        collate_fn = None
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

def get_out_dim(dataset):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'flowers':
        return 102
    elif dataset == 'pets':
        return 37
    elif dataset == 'imdb':
        return 2
    elif dataset == 'imdb':
        return 2
    elif dataset == 'cola':
        return 2
    elif dataset == 'rte':
        return 2
    elif dataset == 'boolq':
        return 2
    elif dataset == 'snli-ve':
        return 3
    elif dataset == 'sst2':
        return 2
    elif dataset == 'mrpc':
        return 2
    elif dataset == 'mnli':
        return 3
    elif dataset == 'wnli':
        return 2
    elif dataset == 'qnli':
        return 2
    elif dataset == 'qqp':
        return 2
    elif dataset == 'aircraft':
        return 100
    elif dataset == 'dtd':
        return 47
    elif dataset == 'food':
        return 101
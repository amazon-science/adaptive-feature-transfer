from itertools import product
import torch
import numpy as np
import random

class MovingAverage:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def update(self, metrics):
        for key, value in metrics.items():
            if key not in self.sums:
                self.sums[key] = 0.0
                self.counts[key] = 0
            self.sums[key] += value
            self.counts[key] += 1

    def average(self):
        averages = {}
        for key in self.sums:
            averages[key] = self.sums[key] / self.counts[key] if self.counts[key] != 0 else 0.0
        return averages

    def reset(self):
        self.sums = {}
        self.counts = {}


def make_grid(hypers):
    # hypers is a dict of lists
    keys = list(hypers.keys())
    for vals in product(*[hypers[k] for k in keys]):
        yield {k: v for k, v in zip(keys, vals)}

def pretty_print_dict(d):
    print('-' * 80)
    for k, v in d.items():
        if isinstance(v, float):
            v = f'{v:.3g}'
        print(f'{k}: {v}')
    print('-' * 80)

def dict_reduce(dicts, reduction='none'):
    # none: dict of lists
    # mean: dict of means
    assert reduction in ['none', 'mean'], f'Invalid reduction: {reduction}'
    out_dict = {}
    # assert same keys
    keys = [set(d.keys()) for d in dicts]
    assert all(k == keys[0] for k in keys), f'Keys not the same: {keys}'
    keys = dicts[0].keys()
    for k in keys:
        vals = [d[k] for d in dicts]
        if reduction == 'none':
            out_dict[k] = vals
        elif reduction == 'mean':
            out_dict[k] = sum(vals) / len(vals)
    return out_dict

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def zero_init(module):
    # zero init for all parameters
    for p in module.parameters():
        p.data.zero_()
    
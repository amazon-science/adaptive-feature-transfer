import torch
import torch.nn as nn
import torch.nn.functional as F

def init_from_pretrained(model, pretrained, freeze_init):
    model_dict = model.state_dict()
    pretrained_dict = pretrained.state_dict()
    model_param_names = list(model_dict.keys())
    not_loading = model_param_names[-2:]
    loading = [name for name in model_param_names if name not in not_loading]
    print(f'Not loading: {not_loading}')
    for name in loading:
        model_dict[name].copy_(pretrained_dict[name])

    if freeze_init:
        # Iterate over the model's parameters and freeze the necessary ones
        for name, param in model.named_parameters():
            if name in loading:
                param.requires_grad = False

def get_param_vec(model, remove_classification_head=False):
    param_list = [p for p in model.parameters()]
    if remove_classification_head:
        param_list = param_list[:-2]
    return torch.flatten(torch.cat([torch.flatten(p) for p in param_list]))

class Prior:
    def __init__(self):
        self.metrics = {}
        self.prec = 0
    def log_prob(self, model):
        raise NotImplementedError
    def train_prior(self, *args, **kwargs):
        return
    def eval_prior(self, model):
        return 0
    def get_test_acc(self):
        return 0
    def pretrain(self, optimizer, steps):
        return
    
class UniformPrior(Prior):
    def __init__(self):
        super().__init__()
    def log_prob(self, *args, **kwargs):
        return 0

class KernelPrior(Prior, nn.Module):
    def __init__(self, feature_dataset, prec, learn_scales=False, tensor_product=False, kernel='linear', diag=True):
        Prior.__init__(self)
        nn.Module.__init__(self)
        self.kernel = kernel
        prior_points = len(feature_dataset)
        self.feat_dims = feature_dataset.feat_dims
        self.tensor_product = tensor_product
        self.prec = prec
        self.diag = diag
        if diag:
            self.s = nn.Parameter(torch.zeros(feature_dataset.num_features))
        else:
            self.s = nn.Parameter(torch.randn(feature_dataset.num_features, feature_dataset.num_features) / (feature_dataset.num_features ** 0.5))
        if not learn_scales:
            self.s.requires_grad = False
        n_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        n_constraints = prior_points ** 2 / 2
        print(f'KernelPrior: params / constraints = {n_params / n_constraints:.2g}')

    def forward(self, pretrained_feat):
        if self.tensor_product:
            # Split features according to feat_dims
            split_features = torch.split(pretrained_feat, self.feat_dims, dim=1)
            # Split scales according to feat_dims
            split_scales = torch.split(self.s.sigmoid(), self.feat_dims, dim=0)
            # Scale each axis
            scaled_features = [f * s for f, s in zip(split_features, split_scales)]
            # Compute tensor product
            scaled_feat = scaled_features[0]
            for i in range(1, len(scaled_features)):
                scaled_feat = torch.einsum('bi,bj->bij', scaled_feat, scaled_features[i]).reshape(scaled_feat.shape[0], -1)
        else:
            if self.diag:
                scaled_feat = pretrained_feat * self.s.sigmoid()
            else:
                scaled_feat = pretrained_feat @ self.s # (B, d) @ (d, d) -> (B, d)
        return scaled_feat


    def get_test_acc(self):
        print('K')
        print(self.k[:5, :5])
        print('target K')
        print(self.k_target[:5, :5])
        print(f'scales: {self.metrics["scales_mean"]:.2f} +- {self.metrics["scales_std"]:.2f}')
        print(f'dk: {self.metrics["dk"]:.3f}')


    def log_prob(self, feat, pretrained_feat):
        target_feat = self(pretrained_feat) # (n, out_dim)
        # center feat and target_feat per dimension
        feat = feat - torch.mean(feat, dim=0, keepdim=True)
        target_feat = target_feat - torch.mean(target_feat, dim=0, keepdim=True)
        # l2 normalize
        feat = feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-8)
        target_feat = target_feat / (torch.norm(target_feat, dim=1, keepdim=True) + 1e-8)

        # compute kernel
        if self.kernel == 'linear':    
            k = torch.matmul(feat, feat.t()) # (n, n)
            k_target = torch.matmul(target_feat, target_feat.t()) # (n, n)
        elif self.kernel == 'rbf':
            # exp(-||x_i - x_j||^2)
            k = torch.exp(- ((feat[:, None, :] - feat[None, :, :]) ** 2).sum(dim=-1)) # (n, n)
            k_target = torch.exp(- ((target_feat[:, None, :] - target_feat[None, :, :]) ** 2).sum(dim=-1)) # (n, n)
        else:
            raise ValueError(f'Unknown kernel {self.kernel}')

        # print first 5 x 5 block
        self.k = k[:5, :5].detach().cpu()
        self.k_target = k_target[:5, :5].detach().cpu()

        dk = k - k_target
        mse = torch.mean(dk ** 2) + 1e-8
        rmse = mse ** 0.5
        self.metrics['dk'] = rmse.item()
        self.metrics['scales_mean'] = torch.mean(self.s.sigmoid()).item()
        self.metrics['scales_std'] = torch.std(self.s.sigmoid()).item()
        return - rmse

class KernelKDPrior(Prior, nn.Module):
    def __init__(self, feature_dataset, feat_dim, prec):
        Prior.__init__(self)
        nn.Module.__init__(self)
        self.kernel = 'linear'
        prior_points = len(feature_dataset)
        self.feat_dims = feature_dataset.feat_dims
        self.prec = prec
        self.proj = nn.Linear(feat_dim, feat_dim)
        n_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        n_constraints = prior_points ** 2 / 2
        print(f'KernelKDPrior: params / constraints = {n_params / n_constraints:.2g}')


    def get_test_acc(self):
        print('K')
        print(self.k[:5, :5])
        print('target K')
        print(self.k_target[:5, :5])
        print(f'dk: {self.metrics["dk"]:.3f}')


    def log_prob(self, feat, pretrained_feat):
        # project downstream features
        target_feat = pretrained_feat # (n, out_dim)
        feat = self.proj(feat) # (n, out_dim)
        # center feat and target_feat per dimension
        feat = feat - torch.mean(feat, dim=0, keepdim=True)
        target_feat = target_feat - torch.mean(target_feat, dim=0, keepdim=True)
        # l2 normalize
        feat = feat / (torch.norm(feat, dim=1, keepdim=True) + 1e-8)
        target_feat = target_feat / (torch.norm(target_feat, dim=1, keepdim=True) + 1e-8)

        # compute kernel
        if self.kernel == 'linear':    
            k = torch.matmul(feat, feat.t()) # (n, n)
            k_target = torch.matmul(target_feat, target_feat.t()) # (n, n)
        elif self.kernel == 'rbf':
            # exp(-||x_i - x_j||^2)
            k = torch.exp(- ((feat[:, None, :] - feat[None, :, :]) ** 2).sum(dim=-1)) # (n, n)
            k_target = torch.exp(- ((target_feat[:, None, :] - target_feat[None, :, :]) ** 2).sum(dim=-1)) # (n, n)
        else:
            raise ValueError(f'Unknown kernel {self.kernel}')

        # print first 5 x 5 block
        self.k = k[:5, :5].detach().cpu()
        self.k_target = k_target[:5, :5].detach().cpu()

        dk = k - k_target
        mse = torch.mean(dk ** 2) + 1e-8
        rmse = mse ** 0.5
        self.metrics['dk'] = rmse.item()
        return - rmse
        

class KDPrior(Prior, nn.Module):
    def __init__(self, feature_dataset, feat_dim, prec, tensor_product=False):
        Prior.__init__(self)
        nn.Module.__init__(self)
        prior_points = len(feature_dataset)
        self.feat_dims = feature_dataset.feat_dims
        self.tensor_product = tensor_product
        self.prec = prec
        if not tensor_product:
            self.proj = nn.Linear(feat_dim, feature_dataset.num_features) # map from downstream features to pretrained features
        else:
            # separate head mapping between features in each modality
            self.proj = nn.ModuleList([nn.Linear(1024, feat_dim) for feat_dim in self.feat_dims])

        n_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        n_constraints = prior_points * feature_dataset.num_features
        print(f'KD: params / constraints = {n_params / n_constraints:.2g}')
    
    def predict_pretrained_feat(self, feat):
        if not self.tensor_product:
            return self.proj(feat)
        else:
            # feat: (vision_feat, text_feat)
            preds = []
            for p, f in zip(self.proj, feat):
                preds.append(p(f))
            return preds

    def log_prob(self, feat, pretrained_feat):
        if not self.tensor_product:
            feat = feat.reshape(feat.shape[0], -1)
            pred = self.predict_pretrained_feat(feat)
            squared_error = (pred - pretrained_feat) ** 2
            return - squared_error.mean()
        else:
            preds = self.predict_pretrained_feat(feat)
            # targets: (vision_feat, text_feat)
            targets = torch.split(pretrained_feat, self.feat_dims, dim=1)
            squared_errors = [torch.mean((p - t) ** 2) for p, t in zip(preds, targets)]
            squared_error = torch.mean(torch.stack(squared_errors))
            return - squared_error

class FeaturePrior(Prior, nn.Module):
    def __init__(self, feature_dataset, feat_dim, prec):
        Prior.__init__(self)
        nn.Module.__init__(self)
        prior_points = len(feature_dataset)
        self.feat_dims = feature_dataset.feat_dims
        self.prec = prec
        self.proj = nn.Linear(feature_dataset.num_features, feat_dim) # map from pretrained features to downstream features
        n_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        n_constraints = prior_points * feature_dataset.num_features
        print(f'FeaturePrior: params / constraints = {n_params / n_constraints:.2g}')


    def log_prob(self, feat, pretrained_feat):    
        feat = feat.reshape(feat.shape[0], -1)
        target_feat = self.proj(pretrained_feat) # (n, out_dim)
        # center feat and target_feat per dimension
        feat = feat - torch.mean(feat, dim=0, keepdim=True)
        target_feat = target_feat - torch.mean(target_feat, dim=0, keepdim=True)
        squared_error = (target_feat - feat) ** 2
        return - squared_error.mean()

class BTunePrior(Prior, nn.Module):
    def __init__(self, feature_dataset, logme_scores, logme_mus, downstream_logme_mu, prec):
        Prior.__init__(self)
        nn.Module.__init__(self)
        self.feat_dims = feature_dataset.feat_dims
        self.logme_scores = torch.tensor(logme_scores)
        self.logme_mus = logme_mus # list of (num_classes, num_features)
        self.logme_mus = [mu.cuda() for mu in logme_mus]
        self.downstream_logme_mu = downstream_logme_mu.cuda() # (num_classes, num_features)
        self.prec = prec
        # softmax over logme_scores at temperature 0.1
        self.logme_weights = torch.softmax(self.logme_scores / 0.1, dim=0).detach().cuda()

    def log_prob(self, feat, pretrained_feat):
        y = torch.matmul(feat, self.downstream_logme_mu.t()) # (n, num_classes)
        with torch.no_grad():
            # chunk the pretrained features
            pretrained_feats = torch.split(pretrained_feat, self.feat_dims, dim=-1)
            y_target = torch.zeros_like(y)
            for i in range(len(pretrained_feats)):
                y_target += self.logme_weights[i] * torch.matmul(pretrained_feats[i], self.logme_mus[i].t())
        mse = torch.mean((y - y_target) ** 2)
        return - mse

class RKdAngle(nn.Module):
    # https://github.com/lenscloth/RKD
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1)) # N x N x C
            norm_td = F.normalize(td, p=2, dim=2) # N x N x C
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1) # N x N x C, N x C x N -> N x N x N

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss

class RkdDistance(nn.Module):
    # https://github.com/lenscloth/RKD
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = torch.cdist(teacher, teacher)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = torch.cdist(student, student)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss
    
class RKDPrior(Prior, nn.Module):
    def __init__(self, feature_dataset, feat_dim, prec):
        Prior.__init__(self)
        nn.Module.__init__(self)
        prior_points = len(feature_dataset)
        self.feat_dims = feature_dataset.feat_dims
        self.prec = prec
        self.angle_loss = RKdAngle()
        self.dist_loss = RkdDistance()

    def log_prob(self, feat, pretrained_feat):
        t_feat = pretrained_feat # (B, *)
        s_feat = feat # (B, *)
        dist_loss = self.dist_loss(s_feat, t_feat)
        angle_loss = self.angle_loss(s_feat, t_feat)
        loss = (dist_loss + 2 * angle_loss) / 2
        return - loss
    
    
class FTPrior(Prior, nn.Module):
    def __init__(self, feature_dataset, feat_dim, prec):
        Prior.__init__(self)
        nn.Module.__init__(self)
        prior_points = len(feature_dataset)
        self.feat_dims = feature_dataset.feat_dims
        self.prec = prec
        # 3 layer MLP
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feature_dataset.num_features),
            nn.ReLU(),
            nn.Linear(feature_dataset.num_features, feature_dataset.num_features)
        )
        # zero init last layer
        self.proj[-1].weight.data.zero_()
        n_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
        n_constraints = prior_points * feature_dataset.num_features
        print(f'FT: params / constraints = {n_params / n_constraints:.2g}')
    
    def predict_pretrained_feat(self, feat):
        return self.proj(feat)

    def log_prob(self, feat, pretrained_feat):
        feat = feat.reshape(feat.shape[0], -1)
        pred = self.predict_pretrained_feat(feat)
        # normalize
        pred = F.normalize(pred, p=2, dim=1)
        pretrained_feat = F.normalize(pretrained_feat, p=2, dim=1)
        l1_loss = (pred - pretrained_feat).abs().sum(-1).mean()
        return - l1_loss
        
@torch.no_grad()
def get_prior(feat_dim, feature_dataset, prec, learn_scales, tensor_product, prior_type):
    assert prior_type in ['kernel', 'kernel_dense', 'kernel_rbf', 'feature', 'kkd', 'kd', 'rkd', 'ft'], f'Unknown prior type {prior_type}'
    if prior_type == 'kernel':
        return KernelPrior(feature_dataset, prec, learn_scales, tensor_product)
    elif prior_type == 'kernel_dense':
        return KernelPrior(feature_dataset, prec, learn_scales, tensor_product, diag=False)
    elif prior_type == 'kernel_rbf':
        return KernelPrior(feature_dataset, prec, learn_scales, tensor_product, kernel='rbf')
    elif prior_type == 'feature':
        assert tensor_product == False, 'tensor_product not supported for feature prior'
        return FeaturePrior(feature_dataset, feat_dim, prec)
    elif prior_type == 'kd':
        if tensor_product:
            assert len(feature_dataset.feat_dims) == 2, 'kd with tensor_product only supported for 2 pretrained features'
            print('Warning: tensor_product for kd assumes the model is CLIP RN50 with features (1024, 1024)')
        return KDPrior(feature_dataset, feat_dim, prec, tensor_product)
    elif prior_type == 'kkd':
        return KernelKDPrior(feature_dataset, feat_dim, prec)
    elif prior_type == 'rkd':
        return RKDPrior(feature_dataset, feat_dim, prec)
    elif prior_type == 'ft':
        return FTPrior(feature_dataset, feat_dim, prec)
    
def get_btune_prior(model_class, dataset, feature_dataset, prec, train_frac):
    feature_paths = feature_dataset.feature_paths
    logme_scores = []
    logme_mus = []
    suffix = '' if train_frac == 1.0 else f'_train_frac={train_frac}'
    for f in feature_paths:
        features = torch.load(f)
        logme_scores.append(features['logme_score' + suffix])
        logme_mus.append(features['logme_weights' + suffix].float())
    downstream_feat_path = f'/fsx/features/{model_class}_{dataset}.pt'
    downstream_logme_mu = torch.load(downstream_feat_path)['logme_weights'  + suffix].float()
    return BTunePrior(feature_dataset, logme_scores, logme_mus, downstream_logme_mu, prec)
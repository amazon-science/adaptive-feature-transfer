import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import transformers
import clip

clip_featdims = {
    'RN50': (1024, 1024),
    'RN101': (512, 512),
    'ViT-B/32': (512, 512),
}

class CLIP(nn.Module):
    def __init__(self, out_dim, version='RN50'):
        super().__init__()
        self.model, self.preprocess = clip.load(version, 'cuda')
        self.model = self.model.float() # make model fp32, or it won't train :(
        dim = clip_featdims[version][0] * clip_featdims[version][1]
        self.feat_dim = dim
        if out_dim == 0:
            self.out = nn.Identity()
        else:
            self.out = nn.Linear(dim, out_dim)
        
    
    def get_transform(self, train):
        def transform(x):
            image = [self.preprocess(img) for img in x['image']]
            text = clip.tokenize(x['hypothesis'])
            return {'image': image, 'text': text, 'label': x['label']}
        return transform
    
    def forward(self, x, return_feat=False):
        # x is a tuple of (image, text) after preprocessing and tokenization
        img, text = x['image'], x['text']
        img_feat = self.model.encode_image(img)
        text_feat = self.model.encode_text(text)
        feat = torch.einsum('bi,bj->bij', img_feat, text_feat).reshape(img_feat.shape[0], -1)
        out = self.out(feat)
        if return_feat:
            return out, feat
        else:
            return out
        
class CLIP_KD(nn.Module):
    def __init__(self, out_dim, version='RN50'):
        super().__init__()
        self.model, self.preprocess = clip.load(version, 'cuda')
        self.model = self.model.float() # make model fp32, or it won't train :(
        dim = clip_featdims[version][0] * clip_featdims[version][1]
        self.feat_dim = dim
        if out_dim == 0:
            self.out = nn.Identity()
        else:
            self.out = nn.Linear(dim, out_dim)
        
    
    def get_transform(self, train):
        def transform(x):
            image = [self.preprocess(img) for img in x['image']]
            text = clip.tokenize(x['hypothesis'])
            return {'image': image, 'text': text, 'label': x['label']}
        return transform
    
    def forward(self, x, return_feat=False):
        # x is a tuple of (image, text) after preprocessing and tokenization
        img, text = x['image'], x['text']
        img_feat = self.model.encode_image(img)
        text_feat = self.model.encode_text(text)
        feat = torch.einsum('bi,bj->bij', img_feat, text_feat).reshape(img_feat.shape[0], -1)
        out = self.out(feat)
        if return_feat:
            return out, (img_feat, text_feat)
        else:
            return out
    
        
class CLIP_RN101(CLIP):
    def __init__(self, out_dim):
        super().__init__(out_dim, version='RN101')

class CLIP_ViTB32(CLIP):
    def __init__(self, out_dim):
        super().__init__(out_dim, version='ViT-B/32')

class LinearModel(nn.Module):
    def __init__(self, num_features, out_dim=10):
        super().__init__()
        self.out = nn.Linear(num_features, out_dim)
        self.feat_dim = num_features
        
        print(f'Parameters: {int(sum([p.numel() for p in self.parameters() if p.requires_grad])/1e3)}k')

    def forward(self, x, return_feat=False):
        out = self.out(x)
        if return_feat:
            return out, x
        else:
            return out
        
class ProductLinearModel(nn.Module):
    def __init__(self, num_features, out_dim=10):
        super().__init__()
        d = num_features // 2
        self.out = nn.Linear(d ** 2, out_dim)
        
        print(f'Parameters: {int(sum([p.numel() for p in self.parameters() if p.requires_grad])/1e3)}k')

    def forward(self, x, return_feat=False):
        x1, x2 = x.chunk(2, dim=1)
        x = torch.einsum('bi,bj->bij', x1, x2).reshape(x1.shape[0], -1)
        out = self.out(x)
        if return_feat:
            return out, x
        else:
            return out

class Concat(nn.Module):
    # a model whose features are the concatenation of the features of two other models
    # the features are also trained
    def __init__(self, models, num_features, out_dim=10):
        super().__init__()
        self.models = nn.ModuleList(models)
        # freeze the models
        for m in self.models:
            m.cuda()
        self.out = nn.Linear(num_features, out_dim)
        
        print(f'Parameters: {int(sum([p.numel() for p in self.parameters()])/1e3)}k')
    
    def forward(self, x):
        feats = []
        for m in self.models:
            feat = m(x)
            feats.append(feat)
        x = torch.cat(feats, dim=1)
        x = self.out(x)
        return x

class TimmWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.classifier = model.get_classifier()
        self.model.reset_classifier(0)
        self.feature_reshape = lambda x: x
        if isinstance(self.classifier, nn.Linear):
            self.feat_dim = self.classifier.in_features
        elif isinstance(self.classifier, nn.Conv2d): # resnetv2
            self.feat_dim = self.classifier.in_channels
            # reshape a flattened feature into 1x1 image
            self.feature_reshape = lambda x: x.reshape(x.shape[0], self.feat_dim, 1, 1)
        else:
            self.feat_dim = None
    
    def forward(self, x, return_feat=False):
        if isinstance(x, dict):
            x = x['image']
        feat = self.model(x)
        feat = self.feature_reshape(feat) # resnetv2
        out = self.classifier(feat)
        out = out.reshape(feat.shape[0], -1) # resnetv2
        feat = feat.reshape(feat.shape[0], -1) # resnetv2
        if return_feat:
            return out, feat
        return out

class HFWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if isinstance(model, transformers.T5ForSequenceClassification):
            self.classifier = model.classification_head.out_proj
            model.classification_head.out_proj = nn.Identity()
        elif isinstance(model, transformers.RobertaForSequenceClassification) or isinstance(model, transformers.XLMRobertaForSequenceClassification):
            self.classifier = model.classifier.out_proj
            model.classifier.out_proj = nn.Identity()
        elif isinstance(model, transformers.DistilBertForSequenceClassification) or isinstance(model, transformers.BertForSequenceClassification) :
            self.classifier = model.classifier
            model.classifier = nn.Identity()
        elif isinstance(model, transformers.GPT2ForSequenceClassification) or isinstance(model, transformers.LlamaForSequenceClassification):
            self.classifier = model.score
            model.score = nn.Identity()
        else:
            raise NotImplementedError(f"Unsupported model type: {type(model)}")
        self.feat_dim = self.classifier.in_features

    
    def forward(self, x, return_feat=False):
        feat = self.model(**x).logits
        out = self.classifier(feat)
        if return_feat:
            return out, feat
        return out
    
class HFFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # a few models performs pooling inside the classifier, so we need to add it back
        self.pooler = lambda x: x
        if isinstance(model, transformers.T5ForSequenceClassification):
            self.classifier = model.classification_head
            model.classification_head = nn.Identity()
        elif isinstance(model, transformers.RobertaForSequenceClassification) or isinstance(model, transformers.XLMRobertaForSequenceClassification):
            self.classifier = model.classifier
            model.classifier = nn.Identity()
            self.pooler = lambda x: x[:, 0, :] # take <s> token (equiv. to [CLS])
        elif isinstance(model, transformers.DistilBertForSequenceClassification) or isinstance(model, transformers.BertForSequenceClassification) :
            self.classifier = model.classifier
            model.classifier = nn.Identity()
        elif isinstance(model, transformers.GPT2ForSequenceClassification) or isinstance(model, transformers.LlamaForSequenceClassification):
            self.classifier = model.score
            model.score = nn.Identity()
        else:
            raise NotImplementedError(f"Unsupported model type: {type(model)}")

    def forward(self, x):
        feat = self.model(**x).logits
        feat = self.pooler(feat)
        return feat

def create_model(model_class, out_dim, pretrained=False, extract_features=False, **kwargs):
    if model_class in globals():
        print("Using custom models, ignoring 'pretrained' argument")
        model = globals()[model_class](out_dim=out_dim, **kwargs)
        get_transform = model.get_transform if hasattr(model, 'get_transform') else None
        tokenizer = None
        input_collate_fn = None
    elif model_class in timm.list_models(pretrained=True):
        model = timm.create_model(model_class, num_classes=out_dim, pretrained=pretrained, **kwargs)
        if pretrained:
            print(f"Using timm pretrained model")
        data_config = timm.data.resolve_model_data_config(model)
        get_transform = lambda train: timm.data.create_transform(**data_config, is_training=train)
        model = TimmWrapper(model)
        tokenizer = None
        input_collate_fn = None
    else:
        if extract_features:
            try:
                model = transformers.AutoModelForSequenceClassification.from_pretrained(model_class, device_map="auto")
            except:
                model = transformers.AutoModelForSequenceClassification.from_pretrained(model_class)
            model = HFFeatureExtractor(model)
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_class, num_labels=out_dim)
            model = HFWrapper(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_class)
        if isinstance(model.model, transformers.GPT2ForSequenceClassification) or isinstance(model.model, transformers.LlamaForSequenceClassification):
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_class, pad_token='<pad>')
            model.model.config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.model_max_length > 10000:
            print(f"NOTE: Truncating tokenizer max length from {tokenizer.model_max_length} to 512")
            tokenizer.model_max_length = 512
        input_collate_fn = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        get_transform = lambda train: None
    return model, get_transform, tokenizer, input_collate_fn
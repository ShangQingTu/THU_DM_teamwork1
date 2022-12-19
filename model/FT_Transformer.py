from typing import Dict, Any
import os
import numpy as np
import pandas as pd
import rtdl
import scipy.special
import sklearn.metrics
from sklearn.model_selection import ShuffleSplit
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import zero

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
# Docs: https://yura52.github.io/delu/0.0.4/reference/api/zero.improve_reproducibility.html
# zero.improve_reproducibility(seed=123456)
class MyResNet(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_tokenizer: rtdl.CategoricalFeatureTokenizer,
        mlp_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.cat_tokenizer = cat_tokenizer
        self.model = rtdl.ResNet.make_baseline(
            d_in=n_num_features + cat_tokenizer.n_tokens * cat_tokenizer.d_token,
            **mlp_kwargs,
        )

    def forward(self, x_num, x_cat):
        return self.model(
            torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
        )

def apply_model(model, x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        return model(x_num, x_cat)

@torch.no_grad()
def evaluate(model, part, num_cat):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], batch_size):
        prediction.append(apply_model(model, batch[:, :-num_cat], batch[:, -num_cat:].long()))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()

    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
        # target = (target > threshold).astype('int')
        # prediction = (prediction > threshold).astype('int')
        # score = (target == prediction).mean()
    return score

def process_categorical(data_frame):
    encoder = sklearn.preprocessing.OneHotEncoder()
    new_feature = []
    for c in data_frame.columns:
        temp = encoder.fit_transform(data_frame[c].values.reshape(-1, 1))
        new_feature.append(np.asarray(temp.argmax(-1)).squeeze())
    return pd.DataFrame(np.array(new_feature).transpose(), columns=data_frame.columns)

def load_dataset(data_dir, use_extra=True):
    X_all = pd.read_csv(f'{data_dir}origin_feature.csv')
    y_all = pd.read_csv(f'{data_dir}label.csv')
    cols = []
    for c in X_all.columns:
        if(len(X_all[c].unique()) == 2):
            cols.append(c)
    X_all['data_channel'] = X_all[cols[:6]].values.argmax(-1)
    X_all['week_day'] = X_all[cols[6:13]].values.argmax(-1)
    X_all = X_all.drop(columns=cols[:13])
    cols = X_all.columns.to_list()
    cols.remove('is_weekend')
    cols.append('is_weekend')
    X_all = X_all[cols] # move categorical features to the end
    if use_extra:
        X_extra = pd.read_csv(f'{data_dir}extra_feature.csv')
        X_extra = process_categorical(X_extra) # change categorical features to [0, # of unique]
        X_all = pd.concat([X_all, X_extra], axis=1)
    X_all = X_all.values
    y_all = y_all.values
    # card = rtdl.data.get_category_sizes(X_all[:, -num_cat:])
    X_cat = X_all[:, -num_cat:]
    card = [np.unique(X_cat[:,i]).shape[0] for i in range(X_cat.shape[1])]
    X = {}
    y = {}
    num = X_all.shape[0]
    X['train'], X['test'], y['train'], y['test'] = X_all[:round(num*0.8), :], X_all[round(num*0.8):, :], y_all[:round(num*0.8)], y_all[round(num*0.8):]
    sp = ShuffleSplit(n_splits=1, test_size=0.2)
    train_index, val_index = next(iter(sp.split(X['train'])))
    X['train'], X['val'], y['train'], y['val'] = X['train'][train_index], X['train'][val_index], y['train'][train_index], y['train'][val_index]
    return X, y, card

def feature_preprocessing(X, y, num_cat, scaler):
    scaler.fit(X['train'][:, :-num_cat])
    X = {
        k: torch.tensor(np.concatenate([scaler.transform(v[:, :-num_cat]), v[:, -num_cat:]], axis=1))
        for k, v in X.items()
    }
    y = {k: torch.tensor(v) for k, v in y.items()}

    X = {
        k: v.float().to(device)
        for k, v in X.items()
    }
    y = {k: v.float().to(device) for k, v in y.items()}

    return X, y

# X_cat = torch.tensor(X_cat.values)
# X_cat_train, X_cat_test = X_cat[:round(num*0.8), :], X_cat[round(num*0.8):, :]
# X_cat_train, X_cat_val = sklearn.model_selection.train_test_split(X_cat_train, train_size=0.8, random_state=42)
data_dir = '/home/cjh/code/DMhomework/THU_DM_teamwork1/data/'
task_type = 'binclass'
num_cat = 7
threshold = 1400
X, y, cat_card = load_dataset(data_dir, use_extra=True)
scaler = sklearn.preprocessing.StandardScaler()
X, y = feature_preprocessing(X, y, num_cat=num_cat, scaler=scaler)
if task_type == 'regression':
    y = {k: v[:, 1] for k, v in y.items()}
    y_mean = y['train'].mean().item()
    y_std = y['train'].std().item()
    y = {k: (v - y_mean) / y_std for k, v in y.items()}
    threshold = (threshold - y_mean) / y_std
elif task_type == 'binclass':
    y = {k: v[:, 0] for k, v in y.items()}
    y_std = y_mean = None
else:
    raise NotImplementedError('No support for muticlass classification!')

# X_cat = X['train'][:, -num_cat:]
# cat_card = [X_cat[:,i].unique().size(0) for i in range(X_cat.size(1))]
# model parameters
d_out = 1
n_block = 3
d_token = 256
ffn_d_hidden = 256
attention_dropout = 0.2
ffn_dropout = 0.2
residual_dropout = 0.0
# training parameters
lr = 1e-5
weight_decay = 0.0
batch_size = 256
n_epochs = 1000
patience = 100


# model = rtdl.FTTransformer.make_baseline(
#     n_num_features=X['train'].shape[1]-num_cat,
#     cat_cardinalities=cat_card,
#     d_token=d_token,
#     n_blocks=n_block,
#     attention_dropout=attention_dropout,
#     ffn_d_hidden=ffn_d_hidden,
#     ffn_dropout=ffn_dropout,
#     residual_dropout=residual_dropout,
#     d_out=d_out
# )

resnet_kwargs = {'n_blocks': n_block, 
                 'd_main': d_token, 
                 'd_hidden': ffn_d_hidden, 
                 'dropout_first': ffn_dropout,
                 'dropout_second': residual_dropout,
                 'd_out': d_out}
model = MyResNet(
    n_num_features=X['train'].shape[1]-num_cat,
    cat_tokenizer=rtdl.CategoricalFeatureTokenizer(cat_card, d_token, 'True', 'uniform'),
    mlp_kwargs=resnet_kwargs
    )

model.to(device)
optimizer = (
    torch.optim.AdamW(model.optimization_param_groups(), lr=lr, weight_decay=weight_decay)
    if isinstance(model, rtdl.FTTransformer)
    else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
)
loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == 'binclass'
    else F.cross_entropy
    if task_type == 'multiclass'
    else F.mse_loss
)
train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)

# Create a progress tracker for early stopping
progress = zero.ProgressTracker(patience=patience)

print(f'Test score before training: {evaluate(model, "test", num_cat):.4f}')

schedular = CosineAnnealingLR(optimizer, T_max=n_epochs)
report_frequency = len(X['train']) // batch_size // 5
for epoch in range(1, n_epochs + 1):
    for iteration, batch_idx in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x_batch = X['train'][batch_idx]
        y_batch = y['train'][batch_idx]
        loss = loss_fn(apply_model(model, x_batch[:, :-num_cat], x_batch[:, -num_cat:].long()).squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')
    schedular.step()
    train_score = evaluate(model, 'train', num_cat)
    val_score = evaluate(model, 'val', num_cat)
    test_score = evaluate(model, 'test', num_cat)
    print(f'Epoch {epoch:03d} | Training score: {train_score:.4f} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
    progress.update((-1 if task_type == 'regression' else 1) * val_score)
    if progress.success:
        print(' <<< BEST VALIDATION EPOCH', end='')
    print()
    if progress.fail:
        break

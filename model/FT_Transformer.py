from typing import Dict, Any
import os
import argparse
import logging
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import rtdl
import scipy.special
from sklearn.model_selection import ShuffleSplit, ParameterGrid, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import zero

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
def evaluate(model, part, num_cat, verbose=False, return_pred=False):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], batch_size):
        prediction.append(apply_model(model, batch[:, :-num_cat], batch[:, -num_cat:].long()))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()
    score = {}
    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
    else:
        assert task_type == 'regression'
        # score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
        target = (target >= threshold).astype('int')
        prediction = (prediction >= threshold).astype('int')
    if verbose:
        score['Accuracy'] = accuracy_score(target, prediction)
        score['AUC_score'] = roc_auc_score(target, prediction)
        score['F1_score'] = f1_score(target, prediction)
    else:
        score['Accuracy'] = accuracy_score(target, prediction)
    return score if not return_pred else prediction

def process_categorical(data_frame):
    encoder = OneHotEncoder()
    new_feature = []
    for c in data_frame.columns:
        temp = encoder.fit_transform(data_frame[c].values.reshape(-1, 1))
        new_feature.append(np.asarray(temp.argmax(-1)).squeeze())
    return pd.DataFrame(np.array(new_feature).transpose(), columns=data_frame.columns)

def load_dataset(data_dir, use_extra=0):
    X_all = pd.read_csv(f'{data_dir}origin_feature.csv')
    y_all = pd.read_csv(f'{data_dir}label.csv')
    cols = []
    for c in X_all.columns:
        if(len(X_all[c].unique()) == 2):
            cols.append(c)
    # change onehot feature to categorical
    X_all['data_channel'] = X_all[cols[:6]].values.argmax(-1) 
    X_all['week_day'] = X_all[cols[6:13]].values.argmax(-1)
    X_all = X_all.drop(columns=cols[:13])
    cols = X_all.columns.to_list()
    # move categorical features to the end
    cols.remove('is_weekend')
    cols.append('is_weekend')
    X_all = X_all[cols] 
    if use_extra:
        X_extra = pd.read_csv(f'{data_dir}extra_feature.csv').iloc[:, :use_extra]
        # change categorical features to [0, cardinality-1]
        X_extra = process_categorical(X_extra) 
        X_all = pd.concat([X_all, X_extra], axis=1)
    X_all = X_all.values
    y_all = y_all.values
    X_cat = X_all[:, -num_cat:]
    # cardinality of categorical feature
    card = [np.unique(X_cat[:,i]).shape[0] for i in range(X_cat.shape[1])] 
    X = {}
    y = {}
    num = X_all.shape[0]
    X['train'], X['test'], y['train'], y['test'] = X_all[:round(num*0.8), :], X_all[round(num*0.8):, :], y_all[:round(num*0.8)], y_all[round(num*0.8):]
    # sp = ShuffleSplit(n_splits=1, test_size=0.2)
    # train_index, test_index = next(iter(sp.split(X_all)))
    # X['train'], X['test'], y['train'], y['test'] = X_all[train_index], X_all[test_index], y_all[train_index], y_all[test_index]
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

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/cjh/code/DMhomework/THU_DM_teamwork1/data/')
    parser.add_argument('--output_dir', default='/home/cjh/code/DMhomework/THU_DM_teamwork1/output/')
    parser.add_argument('--model', choices=['FT-Transformer', 'ResNet'], default='FT-Transformer')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--device_id', default='0')
    parser.add_argument('--ensemble', default=1, type=int)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_dir = args.data_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # logging.basicConfig(filename=output_dir+'log.txt',
    #                     filemode='a',
    #                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    #                     datefmt='%H:%M:%S',
    #                     level=logging.DEBUG)
    task = ['binclass', 'regression'] # solve problem by binary classification or regression
    task_type = task[1]
    num_cat = 3 # number of catgorical feature used
    threshold = 1400
    # return both classification and regression target
    X, y, cat_card = load_dataset(data_dir, use_extra=num_cat-3) 
    scaler = StandardScaler()
    X, y = feature_preprocessing(X, y, num_cat=num_cat, scaler=scaler) # preprocess numerical features
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

    # model parameters
    d_out = 1
    n_block = 4
    d_token = 64
    ffn_d_hidden = 128
    attention_dropout = 0.2
    ffn_dropout = 0.3
    residual_dropout = 0.0
    # training parameters
    lr = 1e-4
    weight_decay = 0.1
    batch_size = 256
    n_epochs = 1000
    patience = 10

    ########################
    # Code for grid search #
    ########################
    # grid = {'n_blocks': [2, 3, 4], 
    #         'd_token': [32, 64, 128],
    #         'ffn_d_hidden': [32, 64, 128],
    #         'attention_dropout': [0.1, 0.2, 0.3],
    #         'ffn_dropout': [0.1, 0.2, 0.3],
    #         'residual_dropout': [0],
    #         }
    # param_grid = ParameterGrid(grid)
    # weight_decays = [1e-2, 1e-1, 1]
    # best_acc = 0.0
    # for param in tqdm(param_grid):
    #     for weight_decay in weight_decays:
    #         model = rtdl.FTTransformer.make_baseline(
    #             n_num_features=X['train'].shape[1]-num_cat, 
    #             cat_cardinalities=cat_card,
    #             d_out=d_out,
    #             **param)

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == 'binclass'
        else F.cross_entropy
        if task_type == 'multiclass'
        else F.mse_loss
    )
    seeds = random.sample(range(1000), args.ensemble)
    for seed in seeds:
        zero.improve_reproducibility(seed=seed)
        if args.model == 'FT-Transformer':
            model = rtdl.FTTransformer.make_baseline(
                n_num_features=X['train'].shape[1]-num_cat,
                cat_cardinalities=cat_card,
                d_token=d_token,
                n_blocks=n_block,
                attention_dropout=attention_dropout,
                ffn_d_hidden=ffn_d_hidden,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
                d_out=d_out
            )
        else:
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
        if not args.eval_only:
            optimizer = (
                AdamW(model.optimization_param_groups(), lr=lr, weight_decay=weight_decay)
                if isinstance(model, rtdl.FTTransformer)
                else AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            )
            train_loader = zero.data.IndexLoader(len(X['train']), batch_size, shuffle=True, device=device)

            # Create a progress tracker for early stopping
            progress = zero.ProgressTracker(patience=patience)
            # print(f"Test score before training: {evaluate(model, 'test', num_cat)['Accuracy']:.4f}")

            # learning rate schedular
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
                print(f"Epoch {epoch:03d} | Training score: {train_score['Accuracy']:.4f} | Validation score: {val_score['Accuracy']:.4f} | Test score: {test_score['Accuracy']:.4f}", end='')
                progress.update(val_score['Accuracy'])
                if progress.success:
                    # Code for grid search
                    # if val_score['Accuracy'] > best_acc:
                    #     best_acc = val_score['Accuracy']
                        # logging.info(f'best parameter: {param} weight_decay: {weight_decay}')
                        # logging.info(f'best validation acc: {best_acc}\n')
                    torch.save(model.state_dict(), f'{output_dir}{model._get_name()}_{seed}.ckpt')
                    print(' <<< BEST VALIDATION EPOCH', end='')
                print()
                if progress.fail:
                    break
    pred = []
    for seed in seeds:
        # if --eval_only choose the checkpoint you have
        model.load_state_dict(torch.load(f'{output_dir}{model._get_name()}_{seed}.ckpt'))
        pred.append(evaluate(model, 'test', num_cat, return_pred=True))
    pred = ((np.array(pred) > 0.5).astype('int').mean(0) > 0.5).astype('int') 
    if task_type == 'binclass':
        target = y['test'].cpu().numpy()
    else:
        target = (y['test'].cpu().numpy() > threshold).astype('int')
    print(f"Test Result:\nAccuracy: {accuracy_score(target, pred)}\nAUC score: {roc_auc_score(target, pred)}\nF1 score: {f1_score(target, pred)}")

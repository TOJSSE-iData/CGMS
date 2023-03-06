import argparse
import sys
import subprocess
import logging
import time
import pickle
import yaml
import random
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as thfn
import dgl
import numpy as np

from datetime import datetime

from torch.utils.data import DataLoader, Subset

from cgms.models import CGMS, AutoEncoder
from cgms.datasets import CGSynDataset, CGSenDataset
from cgms.utils import (
    setup_seed,
    arg_min,
    conf_inv,
    calc_stat,
    save_best_model,
    find_best_model,
    real_random_split_indices
)

time_str = str(datetime.now().strftime('%y%m%d%H%M'))
n_delimiter = 60
logging_fmt = "%(asctime)s %(message)s"
logging_dt_fmt = "[%Y-%m-%d %H:%M:%S]"
DEBUG = False

def step_epoch(model, batch_gs, batch_gs_tail, loader, loader_sen, optimizer, device, train=True, save=False):
    if train:
        model.train()
    else:
        model.eval()
    p_thresh = 0.5
    n_syn_step = len(loader)
    n_total = 0
    n_sen_samples = 0
    loader_iter = iter(loader)
    loader_sen_iter = iter(loader_sen) if loader_sen else None
    accum_loss = 0
    loss_func = nn.MSELoss(reduction='sum')
    y_trues = []
    y_preds = []
    with th.set_grad_enabled(train):
        epoch_loss = 0
        epoch_loss_sen = 0
        while n_syn_step > 0:
            n_total += 1
            p = random.random()
            if not train or p > p_thresh:
                batch = next(loader_iter)
                n_syn_step -= 1
                task_type = 'syn'
            else:
                try:
                    batch = next(loader_sen_iter)
                except StopIteration:
                    loader_sen_iter = iter(loader_sen)
                    batch = next(loader_sen_iter)
                n_sen_samples += batch[0].shape[0]
                task_type = 'sen'
            if device != 'cpu':
                batch = [b.to(device) for b in batch]
            input_feats = {
                'd': batch[0].reshape(batch[0].shape[0]*2, -1),
                'c': batch[1]
            }
            if batch[1].shape[0] == batch_gs['d2c'].num_nodes('c'):
                graphs = batch_gs
            else:
                graphs = batch_gs_tail[task_type]
            labels = batch[2]
            model_outs = model(graphs, input_feats, task_type)
            loss = loss_func(model_outs, labels)
            y_trues.extend(labels.flatten().detach().cpu().numpy().tolist())
            y_preds.extend(model_outs.flatten().detach().cpu().numpy().tolist())
            accum_loss += loss
            if task_type == 'syn':
                epoch_loss += loss.detach()
                if train:
                    optimizer.zero_grad()
                    accum_loss.backward()
                    optimizer.step()
                    accum_loss = 0
            else:
                epoch_loss_sen += loss.detach()
        if DEBUG:
            logging.info(f"{len(loader)} synergy task steps in {n_total} total steps")

        epoch_loss = epoch_loss.item()
        if epoch_loss_sen != 0:
            avg_sen_loss = epoch_loss_sen.item() / n_sen_samples
        else:
            avg_sen_loss = None
        if not train:
            th.cuda.empty_cache()
        if save:
            return epoch_loss, avg_sen_loss, y_trues, y_preds
        return epoch_loss, avg_sen_loss


def predict_epoch_sen(model, batch_gs, batch_gs_tail, loader_sen, device):
    model.eval()
    loss_func = nn.MSELoss(reduction='sum')
    y_trues = []
    y_preds = []
    task_type = 'sen'
    epoch_loss_sen = 0
    with th.no_grad():
        for batch in loader_sen:
            if device != 'cpu':
                batch = [b.to(device) for b in batch]
            input_feats = {
                'd': batch[0].reshape(batch[0].shape[0]*2, -1),
                'c': batch[1]
            }
            if batch[1].shape[0] == batch_gs['d2c'].num_nodes('c'):
                graphs = batch_gs
            else:
                graphs = batch_gs_tail[task_type]
            labels = batch[2]
            model_outs = model(graphs, input_feats, task_type)
            loss = loss_func(model_outs, labels)
            y_trues.extend(labels.flatten().detach().cpu().numpy().tolist())
            y_preds.extend(model_outs.flatten().detach().cpu().numpy().tolist())
            epoch_loss_sen += loss.detach()
        epoch_loss_sen = epoch_loss_sen.item()
        th.cuda.empty_cache()
        return epoch_loss_sen, y_trues, y_preds


def train_model(
    config, model, optimizer, 
    train_set, train_set_sen, valid_set, test_set, test_set_sen,
    batch_size, n_epoch, patience, device, mdl_dir=None
):
    ignored_dl_keys = {'shuffle', 'eval_factor'}
    eval_batch_size = config['dataloader']['eval_factor'] * config['hyper_param']['batch_size']
    dl_config = {
        k: v for k, v in config['dataloader'].items() if k not in ignored_dl_keys
    }
    if valid_set is not None and test_set is not None:
        raise ValueError("One of valid_set and test_set should be None")
    elif valid_set is not None:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **dl_config)
        valid_loader = DataLoader(valid_set, batch_size=eval_batch_size, shuffle=False, **dl_config)
        train_inner = True
        train_size = len(train_set)
        valid_size = len(valid_set)
        test_size = 0
    elif test_set is not None:
        tr_indices, es_indices = real_random_split_indices(len(train_set), test_rate=0.1)
        tr_subset = Subset(train_set, tr_indices)
        es_subset = Subset(train_set, es_indices)
        train_loader = DataLoader(tr_subset, batch_size=batch_size, shuffle=True, **dl_config)
        valid_loader = DataLoader(es_subset, batch_size=eval_batch_size, shuffle=False, **dl_config)
        test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, **dl_config)
        train_inner = False
        train_size = len(tr_indices)
        valid_size = len(es_indices)
        test_size = len(test_set)
        test_sen_loader = DataLoader(test_set_sen, batch_size=eval_batch_size, shuffle=False, **dl_config)
    else:
        raise AssertionError("Should never reach here.")
    train_sen_loader = DataLoader(train_set_sen, batch_size=batch_size, shuffle=True, **dl_config)
    train_size_sen = len(train_set_sen)

    batch_gs_train = create_batch_graph(batch_size, device)
    batch_gs_tail_train = {
        'syn': create_batch_graph(train_size % batch_size, device),
        'sen': create_batch_graph(train_size_sen % batch_size, device)
    }
    # bgl_tr
    batch_gs_eval = create_batch_graph(eval_batch_size, device)
    batch_gs_tail_valid = {'syn': create_batch_graph(valid_size % eval_batch_size, device)}
    batch_gs_tail_test = {
        'syn': create_batch_graph(test_size % eval_batch_size, device),
    }
    if test_set_sen is not None:
        batch_gs_tail_test['sen'] = create_batch_graph(len(test_set_sen) % eval_batch_size, device)

    loss_value = float('inf')
    angry = 0
    for epoch in range(1, n_epoch + 1):
        trn_loss, trn_loss_sen = step_epoch(
            model, batch_gs_train, batch_gs_tail_train, 
            train_loader, train_sen_loader, optimizer, device, train=True
        )
        trn_loss /= train_size
        val_loss, _ = step_epoch(
            model, batch_gs_eval, batch_gs_tail_valid,
            valid_loader, None, optimizer, device, train=False
        )
        val_loss /= valid_size
        if DEBUG:
            logging.info(f"Epoch {epoch} | Train loss: {trn_loss}")
            logging.info(f"Epoch {epoch} | Train Sen loss: {trn_loss_sen}")
            logging.info(f"Epoch {epoch} | Valid loss: {val_loss}")
        if val_loss < loss_value:
            angry = 0
            loss_value = val_loss
            if not train_inner:
                save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)
        else:
            angry += 1
            if angry >= patience:
                break
    if not train_inner:
        model.load_state_dict(th.load(find_best_model(mdl_dir)))
        test_loss, _, y_trues, y_preds = step_epoch(
            model, batch_gs_eval, batch_gs_tail_test,
            test_loader, None, None, device, train=False, save=True
        )
        loss_value = test_loss / test_size
        with open(os.path.join(mdl_dir, 'y_trues.pkl'), 'wb') as f:
            pickle.dump(y_trues, f)
        with open(os.path.join(mdl_dir, 'y_preds.pkl'), 'wb') as f:
            pickle.dump(y_preds, f)

        test_sen_loss, y_trues, y_preds = predict_epoch_sen(
            model, batch_gs_eval, batch_gs_tail_test, test_sen_loader, device
        )
        loss_value_sen = test_sen_loss / len(test_set_sen)
        with open(os.path.join(mdl_dir, 'y_trues_sen.pkl'), 'wb') as f:
            pickle.dump(y_trues, f)
        with open(os.path.join(mdl_dir, 'y_preds_sen.pkl'), 'wb') as f:
            pickle.dump(y_preds, f)
        th.cuda.empty_cache()
    else:
        logging.info(f"Train Sen Loss: {trn_loss_sen}")
        loss_value_sen = trn_loss_sen
    return loss_value, loss_value_sen


def create_model(train_set, hid_dims):
    drug_dim = train_set.drug_feats.shape[1]
    cell_dim = train_set.cell_feats.shape[1]
    model = CGMS(drug_dim, cell_dim, hid_dims)
    return model


def create_batch_graph(bs, device):
    if bs <= 0:
        return None
    gs = {}
    for mp in CGMS.get_metapaths()[:-1]:
        gs[mp] = dgl.batch([CGMS.get_graph_by_metapath(mp, 2) for _ in range(bs)]).to(device)
    return gs


def cv_fold(config, test_fold):
    hyper_params = config['hyper_param']
    if config['gpu'] >= 0 and th.cuda.is_available():
        device = f"cuda:{config['gpu']}"
    else:
        device = 'cpu'

    outer_trn_folds = [x for x in config['folds'] if x != test_fold]
    logging.info(f"Outer: train folds {outer_trn_folds}, test folds {test_fold}")
    logging.info("-" * n_delimiter)
    if config['task'] in ('debug', 'exp'):
        best_hs = hyper_params['hidden'][0]
        best_lr = float(hyper_params['lr'][0])
    else:
        param = []
        losses = []
        for hs in hyper_params['hidden']:
            for lr in hyper_params['lr']:
                lr = float(lr)
                param.append((hs, lr))
                logging.info(f"Hidden size: {hs} | Learning rate: {lr}")
                ret_vals = []
                for valid_fold in outer_trn_folds:
                    inner_trn_folds = [x for x in outer_trn_folds if x != valid_fold]
                    valid_folds = [valid_fold]
                    train_set_syn = CGSynDataset(
                        config['drug_feat'], config['cell_feat'],
                        config['synergy'], use_folds=inner_trn_folds
                    )
                    valid_set_syn = CGSynDataset(
                        config['drug_feat'], config['cell_feat'],
                        config['synergy'], use_folds=valid_folds
                    )
                    train_set_sen = CGSenDataset(
                        config['drug_feat'], config['cell_feat'],
                        config['sensitivity'], use_folds=outer_trn_folds
                    )
                    model = create_model(train_set_syn, hs)
                    model = model.to(device)
                    optimizer = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
                    logging.info(f"Start inner loop: train folds {inner_trn_folds}, valid folds {valid_folds}")
                    ret, _ = train_model(
                        config, model, optimizer, train_set_syn, train_set_sen, valid_set_syn, None, None,
                        hyper_params['batch_size'], hyper_params['epoch'], hyper_params['patience'],
                        device, mdl_dir=None
                    )
                    ret_vals.append(ret)
                    del model
                inner_loss = sum(ret_vals) / len(ret_vals)
                logging.info(f"Inner loop completed. Mean valid loss: {inner_loss:.4f}")
                logging.info("-" * n_delimiter)
                losses.append(inner_loss)
        th.cuda.empty_cache()
        time.sleep(10)
        min_ls, min_idx = arg_min(losses)
        best_hs, best_lr = param[min_idx]
    train_set_syn = CGSynDataset(
        config['drug_feat'], config['cell_feat'],
        config['synergy'], use_folds=outer_trn_folds
    )
    train_set_sen = CGSenDataset(
        config['drug_feat'], config['cell_feat'],
        config['sensitivity'], use_folds=outer_trn_folds
    )
    test_set = CGSynDataset(
       config['drug_feat'], config['cell_feat'],
        config['synergy'], use_folds=[test_fold]
    )
    test_set_sen = CGSenDataset(
        config['drug_feat'], config['cell_feat'],
        config['sensitivity'], use_folds=[test_fold]
    )
    model = create_model(train_set_syn, best_hs)
    model = model.to(device)
    optimizer = th.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=0.01)

    logging.info(f"Best hidden size: {best_hs} | Best learning rate: {best_lr}")
    logging.info(f"Start test on fold {[test_fold]}.")
    test_mdl_dir = os.path.join(out_dir, str(test_fold))
    if not os.path.exists(test_mdl_dir):
        os.makedirs(test_mdl_dir)
    test_loss, test_loss_sen = train_model(
        config, model, optimizer, train_set_syn, train_set_sen, None, test_set, test_set_sen,
        hyper_params['batch_size'], hyper_params['epoch'], hyper_params['patience'],
        device, mdl_dir=test_mdl_dir
    )

    with open(os.path.join(test_mdl_dir, 'y_trues_sen.pkl'), 'rb') as f:
        yts = pickle.load(f)
    with open(os.path.join(test_mdl_dir, 'y_preds_sen.pkl'), 'rb') as f:
        yps = pickle.load(f)
    pcc = np.corrcoef(yts, yps)[0, 1]
    logging.info(f"Test sen  pcc: {pcc:.4f}")
    logging.info(f"Test sen rmse: {test_loss_sen**0.5:.4f}")
    logging.info(f"Test sen  mse: {test_loss_sen:.4f}")

    with open(os.path.join(test_mdl_dir, 'y_trues.pkl'), 'rb') as f:
        yts = pickle.load(f)
    with open(os.path.join(test_mdl_dir, 'y_preds.pkl'), 'rb') as f:
        yps = pickle.load(f)
    pcc = np.corrcoef(yts, yps)[0, 1]
    logging.info(f"Test  pcc: {pcc:.4f}")
    logging.info(f"Test rmse: {test_loss**0.5:.4f}")
    logging.info(f"Test  mse: {test_loss:.4f}")


def cv_all(config_fp, calc_metric_only=False):
    processes = []
    final_cv_fn = os.path.join(out_dir, 'cv.log')
    out_f = open(final_cv_fn, 'w')
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} CV start\n")
    if calc_metric_only:
        out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} Calculate metric only. Train processes have finished before.\n")
    else:
        for test_fold in config['folds']:
            cmd = [
                sys.executable, 'cv_cgms.py', config_fp,
                '--fold', str(test_fold)
            ]
            out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} add subprocess: {' '.join(cmd)}\n")
            process = subprocess.Popen(cmd)
            processes.append(process)

    if not calc_metric_only:
        process_codes = [p.wait() for p in processes]
        assert sum([x ** 2 for x in process_codes]) == 0
    
    mses = []
    y_trues = []
    y_preds = []
    pearson_coefs = []

    mses_sen = []
    y_trues_sen = []
    y_preds_sen = []
    pearson_coefs_sen = []
    for test_fold in config['folds']:
        out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} {'$' * n_delimiter}\n\n")

        with open(os.path.join(out_dir, f'fold{test_fold}.log'), 'r') as f:
            lines = [l for l in f.readlines() if len(l) > 0]
            for line in lines[-6:]:
                out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} {line}")

        with open(os.path.join(out_dir, f'{test_fold}', 'y_trues.pkl'), 'rb') as f:
            yts = pickle.load(f)
            y_trues += yts
        with open(os.path.join(out_dir, f'{test_fold}', 'y_preds.pkl'), 'rb') as f:
            yps = pickle.load(f)
            y_preds += yps
        se = 0
        for (y, yy) in zip(yts, yps):
            se += (y-yy)**2
        se /= len(yps)
        mses.append(se)
        pcc = np.corrcoef(yts, yps)[0, 1]
        pearson_coefs.append(pcc)

        with open(os.path.join(out_dir, f'{test_fold}', 'y_trues_sen.pkl'), 'rb') as f:
            yts = pickle.load(f)
            y_trues_sen += yts
        with open(os.path.join(out_dir, f'{test_fold}', 'y_preds_sen.pkl'), 'rb') as f:
            yps = pickle.load(f)
            y_preds_sen += yps
        se = 0
        for (y, yy) in zip(yts, yps):
            se += (y-yy)**2
        se /= len(yps)
        mses_sen.append(se)
        pcc = np.corrcoef(yts, yps)[0, 1]
        pearson_coefs_sen.append(pcc)
    
    mu, sigma = calc_stat(mses)
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} MSE: {mu:.4f} ± {sigma:.4f}\n")
    rmse_loss = [x ** 0.5 for x in mses]
    mu, sigma = calc_stat(rmse_loss)
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} RMSE: {mu:.4f} ± {sigma:.4f}\n")
    mu, sigma = calc_stat(pearson_coefs)
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} PCC: {mu:.4f} ± {sigma:.4f}\n")
    mu, sigma = calc_stat(mses)
    lo, hi = conf_inv(mu, sigma, len(mses))
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} Confidence interval: [{lo:.4f}, {hi:.4f}]\n")

    mu, sigma = calc_stat(mses_sen)
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} MSE Sen: {mu:.4f} ± {sigma:.4f}\n")
    rmse_loss = [x ** 0.5 for x in mses_sen]
    mu, sigma = calc_stat(rmse_loss)
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} RMSE Sen: {mu:.4f} ± {sigma:.4f}\n")
    mu, sigma = calc_stat(pearson_coefs_sen)
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} PCC Sen: {mu:.4f} ± {sigma:.4f}\n")
    mu, sigma = calc_stat(mses_sen)
    lo, hi = conf_inv(mu, sigma, len(mses_sen))
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} Confidence interval Sen: [{lo:.4f}, {hi:.4f}]\n")
    out_f.write(f"{datetime.now().strftime(logging_dt_fmt)} CV completed")
    out_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="config file path")
    parser.add_argument('--fold', type=int, default=-1, help="cv on specified fold")
    parser.add_argument('--seed', type=int, default=datetime.now().microsecond % 1024, help='random seed')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    parser.add_argument('--metric', action='store_true', help='calc metric only')
    parser.add_argument('--suffix', type=str, default='cgms', help='suffix')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as yaml_file:
        config = yaml.safe_load(yaml_file)
    if 'random_seed' not in config.keys():
        config['random_seed'] = args.seed
    if 'gpu' not in config.keys():
        config['gpu'] = args.gpu
    if 'suffix' not in config.keys():
        config['suffix'] = args.suffix
    out_dir = os.path.join('output', f"{config['task']}_{config['suffix']}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    setup_seed(config['random_seed'])
    DEBUG = config['task'] == 'debug'

    if args.fold < 0:
        # cv on all
        cur_config_fp = os.path.join(out_dir, 'config.yaml')
        with open(cur_config_fp, 'w', encoding='UTF-8') as yaml_file:
            yaml.safe_dump(config, yaml_file)
        cv_all(cur_config_fp, args.metric)
    else:
        # cv on specified fold
        log_file = os.path.join(out_dir, f'fold{args.fold}.log')
        logging.basicConfig(
            filename=log_file,
            format=logging_fmt,
            datefmt=logging_dt_fmt,
            level=logging.INFO
        )
        cv_fold(config, args.fold)
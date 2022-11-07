import argparse
import logging
import time
import pickle
# from line_profiler import LineProfiler
import random
import os
import torch as th
import torch.nn as nn

from datetime import datetime

from torch.utils.data import DataLoader, SubsetRandomSampler

from constant import HYPERPARAMETERS, EPOCHS
from cgm.models import MultiTaskModel
from cgm.datasets import SynergyDataset, SensitivityDataset
from cgm.utils import random_split_indices, save_best_model, find_best_model, arg_min, calc_stat, conf_inv

time_str = str(datetime.now().strftime('%y%m%d%H%M'))


def step_epoch(model, loader, loader_sen, optimizer, device, train=True):
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
    with th.set_grad_enabled(train):
        epoch_loss = 0
        epoch_loss_sen = 0
        while n_syn_step > 0:
            do_bp = False
            n_total += 1
            p = random.random()
            if not train or p > p_thresh:
                batch = next(loader_iter)
                n_syn_step -= 1
                use_syn = True
                do_bp = True
            else:
                try:
                    batch = next(loader_sen_iter)
                except StopIteration:
                    loader_sen_iter = iter(loader_sen)
                    batch = next(loader_sen_iter)
                n_sen_samples += batch[0].shape[0]
                use_syn = False
            if not use_syn:
                batch = list(batch)
                batch[1] = None
            new_batch = []
            for b in batch:
                if b is not None:
                    new_batch.append(b.to(device))
                else:
                    new_batch.append(None)
            drug1, drug2, cell, reg_lbls, _ = new_batch
            if train:
                model_outs = model(drug1, drug2, cell)
            else:
                model_outs1 = model(drug1, drug2, cell)
                model_outs2 = model(drug2, drug1, cell)
                model_outs = []
                for o1, o2 in zip(model_outs1, model_outs2):
                    if o1 is not None:
                        model_outs.append((o1 + o2) / 2)
                    else:
                        model_outs.append(None)
            total_loss, syn_loss, sen_loss = model.loss_func(model_outs, reg_lbls)
            accum_loss += total_loss
            if train and do_bp:
                optimizer.zero_grad()
                accum_loss.backward()
                optimizer.step()
                accum_loss = 0

            if not isinstance(syn_loss, int):
                epoch_loss += syn_loss.detach()  # synergy reg loss
            else:
                epoch_loss_sen += sen_loss.detach()
        logging.debug(f"{len(loader)} synergy task steps in {n_total} total steps")

        epoch_loss = epoch_loss.item()
        if epoch_loss_sen != 0:
            avg_sen_loss = epoch_loss_sen.item() / n_sen_samples
        else:
            avg_sen_loss = None
        return epoch_loss, avg_sen_loss


def train_model(model, optimizer, train_set, train_set_sen, valid_set, test_set, batch_size, n_epoch, patience,
                device, mdl_dir=None):
    # lp = LineProfiler()
    # lp_wrapper = lp(step_epoch)
    if valid_set is not None and test_set is not None:
        raise ValueError("One of valid_set and test_set should be None")
    elif valid_set is not None:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        train = True
        train_size = len(train_set)
        valid_size = len(valid_set)
        test_size = 0
    elif test_set is not None:
        tr_indices, es_indices = random_split_indices(len(train_set), test_rate=0.1)
        tr_sampler = SubsetRandomSampler(tr_indices)
        es_sampler = SubsetRandomSampler(es_indices)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=tr_sampler)
        valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=es_sampler)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        train = False
        train_size = len(tr_indices)
        valid_size = len(es_indices)
        test_size = len(test_set)
    else:
        raise AssertionError("Should never reach here.")
    train_sen_loader = DataLoader(train_set_sen, batch_size=batch_size, shuffle=True)

    loss_value = float('inf')
    angry = 0
    for epoch in range(1, n_epoch + 1):
        trn_loss, trn_loss_sen = step_epoch(model, train_loader, train_sen_loader, optimizer, device, train=True)
        trn_loss /= train_size
        val_loss, _ = step_epoch(model, valid_loader, None, optimizer, device, train=False)
        val_loss /= valid_size
        logging.debug(f"Epoch {epoch} | Train loss: {trn_loss}")
        logging.debug(f"Epoch {epoch} | Train Sen loss: {trn_loss_sen}")
        logging.debug(f"Epoch {epoch} | Valid loss: {val_loss}")
        if val_loss < loss_value:
            angry = 0
            loss_value = val_loss
            if not train:
                save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)
        else:
            angry += 1
            if angry >= patience:
                break
    if not train:
        model.load_state_dict(th.load(find_best_model(mdl_dir)))
        test_loss, _ = step_epoch(model, test_loader, None, None, device, train=False)
        loss_value = test_loss / test_size
    logging.info(f"Train Sen Loss: {trn_loss_sen}")
    # lp.print_stats()
    return loss_value


def create_model(train_set, hid_dims, dropout, device):
    drug_dim = train_set.drug_feats.shape[1]
    cell_dim = train_set.cell_feats.shape[1]
    model = MultiTaskModel(drug_dim, cell_dim, hid_dims, dropout)
    return model.to(device)


def cv(args, out_dir):
    test_loss_file = os.path.join(out_dir, 'test_loss.pkl')

    device = f'cuda:{args.gpu}' if (th.cuda.is_available() and args.gpu is not None) else 'cpu'

    n_folds = 5
    n_delimiter = 60
    test_losses = []
    test_folds = [args.fold] if args.fold is not None else range(n_folds)
    for test_fold in test_folds:
        outer_trn_folds = [x for x in range(n_folds) if x != test_fold]
        logging.info("Outer: train folds {}, test folds {}".format(outer_trn_folds, test_fold))
        logging.info("-" * n_delimiter)
        losses = []
        if args.nest:
            for hyper_param in HYPERPARAMETERS:
                logging.info(f"Hidden size: {hyper_param['hidden_dims']} | Learning rate: {hyper_param['lr']}")
                ret_vals = []
                for valid_fold in outer_trn_folds:
                    inner_trn_folds = [x for x in outer_trn_folds if x != valid_fold]
                    valid_folds = [valid_fold]
                    train_set = SynergyDataset(
                        'data/proc/syn4cv.csv',
                        'data/proc/drug_feat_ae.npy',
                        'data/proc/cell_feat_ae.npy',
                        inner_trn_folds,
                        double=True
                    )
                    valid_set = SynergyDataset(
                        'data/proc/syn4cv.csv',
                        'data/proc/drug_feat_ae.npy',
                        'data/proc/cell_feat_ae.npy',
                        valid_folds
                    )
                    train_set_sen = SensitivityDataset(
                        'data/proc/sen_syn_feat_dcnt5.csv',
                        'data/proc/drug_feat_ae.npy',
                        'data/proc/cell_feat_ae.npy',
                        inner_trn_folds
                    )
                    logging.debug(f"train set size: {train_set.ddc.shape}")
                    logging.debug(f"valid set size: {valid_set.ddc.shape}")
                    model = create_model(train_set, hyper_param['hidden_dims'], hyper_param['drop_out'], device)
                    optimizer = th.optim.Adam(model.parameters(), lr=hyper_param['lr'])
                    logging.info(
                        "Start inner loop: train folds {}, valid folds {}".format(inner_trn_folds, valid_folds))
                    ret = train_model(
                        model, optimizer, train_set, train_set_sen, valid_set, None, hyper_param['batch_size'],
                        EPOCHS, args.patience, device, mdl_dir=None
                    )
                    ret_vals.append(ret)
                    del model

                inner_loss = sum(ret_vals) / len(ret_vals)
                logging.info("Inner loop completed. Mean valid loss: {:.4f}".format(inner_loss))
                logging.info("-" * n_delimiter)
                losses.append(inner_loss)
            th.cuda.empty_cache()
            time.sleep(10)
            min_ls, min_idx = arg_min(losses)
            best_hp = HYPERPARAMETERS[min_idx]
        else:
            best_hp = HYPERPARAMETERS[1]
        train_set = SynergyDataset(
            'data/proc/syn4cv.csv',
            'data/proc/drug_feat_ae.npy',
            'data/proc/cell_feat_ae.npy',
            outer_trn_folds,
            double=True
        )
        test_set = SynergyDataset(
            'data/proc/syn4cv.csv',
            'data/proc/drug_feat_ae.npy',
            'data/proc/cell_feat_ae.npy',
            [test_fold]
        )
        train_set_sen = SensitivityDataset(
            'data/proc/sen_syn_feat_dcnt5.csv',
            'data/proc/drug_feat_ae.npy',
            'data/proc/cell_feat_ae.npy',
            outer_trn_folds
        )
        test_set_sen = SensitivityDataset(
            'data/proc/sen_syn_feat_dcnt5.csv',
            'data/proc/drug_feat_ae.npy',
            'data/proc/cell_feat_ae.npy',
            [test_fold]
        )
        model = create_model(train_set, best_hp['hidden_dims'], best_hp['drop_out'], device)
        optimizer = th.optim.Adam(model.parameters(), lr=best_hp['lr'])

        logging.info(f"Best hidden size: {best_hp['hidden_dims']} | Best learning rate: {best_hp['lr']}")
        logging.info("Start test on fold {}.".format([test_fold]))
        test_mdl_dir = os.path.join(out_dir, str(test_fold))
        if not os.path.exists(test_mdl_dir):
            os.makedirs(test_mdl_dir)
        test_loss = train_model(model, optimizer, train_set, train_set_sen, None, test_set, best_hp['batch_size'],
                                EPOCHS, args.patience, device, mdl_dir=test_mdl_dir)

        # sen loss
        with th.no_grad():
            test_loss_sen = 0
            test_loader_sen = DataLoader(test_set_sen, batch_size=best_hp['batch_size'], shuffle=False)
            for batch in test_loader_sen:
                drug1, drug2, cell, reg_lbls, _ = batch
                model_outs1 = model(drug1.to(device), None, cell.to(device))
                _, _, sen_loss = model.loss_func(model_outs1, reg_lbls.to(device))
                test_loss_sen += sen_loss.item()

        test_losses.append(test_loss)
        logging.info(f"Test loss: {test_loss:.4f}")
        logging.info(f"Test Sen loss: {test_loss_sen / len(test_set_sen):.4f}")
        logging.info("*" * n_delimiter + '\n')
    logging.info("CV completed")
    with open(test_loss_file, 'wb') as f:
        pickle.dump(test_losses, f)
    mu, sigma = calc_stat(test_losses)
    logging.info("MSE: {:.4f} ± {:.4f}".format(mu, sigma))
    lo, hi = conf_inv(mu, sigma, len(test_losses))
    logging.info("Confidence interval: [{:.4f}, {:.4f}]".format(lo, hi))
    rmse_loss = [x ** 0.5 for x in test_losses]
    mu, sigma = calc_stat(rmse_loss)
    logging.info("RMSE: {:.4f} ± {:.4f}".format(mu, sigma))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nest', action='store_true')
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=None,
                        help="gpu id")
    parser.add_argument('--patience', type=int, default=20,
                        help='patience for early stop')
    parser.add_argument('--suffix', type=str, default=time_str,
                        help="model dir suffix")
    args = parser.parse_args()
    out_dir = os.path.join('output', 'cv_mtl_{}'.format(args.suffix))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file = os.path.join(out_dir, 'cv.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)

    cv(args, out_dir)


if __name__ == '__main__':
    main()

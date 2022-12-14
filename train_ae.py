import argparse
import torch as th
import os

from datetime import datetime

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from cgms.models import AutoEncoder
from cgms.datasets import AEDataset
from cgms.utils import save_best_model, find_best_model, save_args


def main(args):
    if args.ae_type == 0:
        entity = 'drug'
    else:
        entity = 'cell'
    ae_ds = AEDataset(f'./data/raw_{entity}_feat.npy')
    out_fn = f'{entity}_feat_ae.npy'
    output_dir = 'output'
    output_entity_dir = os.path.join(output_dir, f'{entity}_ae')
    if not os.path.exists(output_entity_dir):
        os.makedirs(output_entity_dir)
    save_args(args, os.path.join(output_entity_dir, 'args.json'))

    trn_loader = DataLoader(ae_ds, batch_size=len(ae_ds), shuffle=True)
    test_loader = DataLoader(ae_ds, batch_size=len(ae_ds), shuffle=False)

    model = AutoEncoder(ae_ds.data.shape[1], [args.dim * 2, args.dim])
    if th.cuda.is_available():
        model = model.to('cuda')
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for i in range(1, args.epoch + 1):
        model.train()
        for x in trn_loader:
            if th.cuda.is_available():
                x = x.to('cuda')
            model_outs = model(x)
            trn_loss = model.calc_loss(x, model_outs)
            optimizer.zero_grad()
            trn_loss.backward()
            optimizer.step()
            print(f"Epoch {i:03d} | Train Loss: {trn_loss.item():.6f}")
            save_best_model(model.state_dict(), output_entity_dir, i, keep=2)
    with th.no_grad():
        model.eval()
        model.load_state_dict(th.load(find_best_model(output_entity_dir)))
        for x in test_loader:
            if th.cuda.is_available():
                x = x.to('cuda')
            # x_hat, mu, log_var, z = model(x)
            z, x_hat = model(x)
            out = z.cpu().numpy()
            print(f"mean diff: {F.l1_loss(x_hat, x, reduction='mean') :.6f}")
            np.save(os.path.join(output_dir, out_fn), out)
            np.save(os.path.join(output_entity_dir, f'reconstruct_x_{entity}.npy'), x_hat.cpu().numpy())


if __name__ == '__main__':
    time_str = str(datetime.now().strftime('%y%m%d%H%M'))
    parser = argparse.ArgumentParser('train ae')
    parser.add_argument('ae_type', type=int, choices=[0, 1], help='0 for drug, 1 for cell')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
"""
NOAA temperature interpolation: NP vs DNP.
Mirrors run_isuper_comparison.py structure but uses NoaaTemperatureDataset.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import random
import ast
from torch.distributions import Normal
from pathlib import Path
from torch.utils.data import DataLoader
from torch.distributions.kl import kl_divergence

from neural_process import NPregression
from utils import chosen_context_target_split
from dnp_multiple import DenoisingNeuralProcessMultiNoise
from noaa_dataset import NoaaTemperatureDataset


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CONFIGURATIONS
# Each entry defines a self-contained experiment.
#   xdim              : number of input features
#   col_indices       : columns to keep from the (lat, lon, elevation) order
#   noise_std_per_dim : per-dim noise std for DNP (normalized scale)
#   denoise_dims      : which dims DNP will denoise
#
# Elevation noise σ=10m → normalized 10/elev_range. With CA-Q1 2026 data,
# elev range = 2174m → σ_norm ≈ 0.0046.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_CONFIGS = {
    'lat_lon_temp': {
        'xdim': 3,
        'col_indices': [0, 1, 2],
        # σ_temp = 0.5°C / 27.1°C range = 0.01845
        'noise_std_per_dim': {2: 0.01845},
        'denoise_dims': [2],
    },
    'lat_lon_rhum': {
        'xdim': 3,
        'col_indices': [0, 1, 3],   # lat, lon, rhum (skip temp)
        # σ_rhum = 2% RH / 87% range = 0.02299
        # NOTE: after slicing, rhum lands at index 2
        'noise_std_per_dim': {2: 0.02299},
        'denoise_dims': [2],
    },
    'lat_lon_temp_rhum': {
        'xdim': 4,
        'col_indices': [0, 1, 2, 3],
        'noise_std_per_dim': {2: 0.01845, 3: 0.02299},
        'denoise_dims': [2, 3],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    'seed':         0,
    'data_dir':     'data/noaa/noaa_california_2026-Q1',
    'hdim':         128,
    'batch_size':   16,
    'epochs':       200,
    'lr':           1e-3,
    'beta':         1.0,   
    'save_dir':     'results/noaa_comparison',
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits(set_path):
    """Parse context / target / test sets — same format as ISUPER."""
    splits = {}
    with open(set_path) as f:
        for line in f:
            for key in ('context_set', 'target_set', 'test_set'):
                if line.startswith(key):
                    splits[key] = ast.literal_eval(line.split('=', 1)[1])
    return splits


class SlicedNoaaDataset(torch.utils.data.Dataset):
    """Wrap NoaaTemperatureDataset and select columns for a feature config."""
    def __init__(self, base, col_indices):
        self.base = base
        self.cols = col_indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x[..., self.cols], y


def make_loader(data_dir, split, feat_cfg, batch_size, train_norm_stats=None):
    """
    Build a DataLoader for a NOAA split.
    Always loads the full (lat, lon, elevation) features from the saved CSV;
    SlicedNoaaDataset selects the columns the feature config wants.

    Returns (loader, norm_stats). For 'train', call with train_norm_stats=None
    so stats are computed; pass the returned stats to val/test loaders.
    """
    base_ds = NoaaTemperatureDataset(
        data_dir, split=split, norm_stats=train_norm_stats
    )
    cols = feat_cfg['col_indices']
    ds = base_ds if cols == [0, 1, 2, 3] else SlicedNoaaDataset(base_ds, cols)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'))
    return loader, base_ds.norm_stats


# ─────────────────────────────────────────────────────────────────────────────
# DNP WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
class DenoisingNP(nn.Module):
    """Same wrapper as ISUPER — handles per-dim noise + selective denoising."""
    def __init__(self, xdim, ydim=1, hdim=128,
                 noise_std_per_dim=None, denoise_dims=None, denoise_uses_y=False):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.denoise_dims = denoise_dims if denoise_dims else []
        self.noise_std_per_dim_dict = noise_std_per_dim if noise_std_per_dim else {}

        noise_std_array = [
            self.noise_std_per_dim_dict.get(i, 0.0) for i in range(xdim)
        ]

        self.dnp = DenoisingNeuralProcessMultiNoise(
            x_dim=xdim, y_dim=ydim,
            r_dim=hdim, z_dim=hdim, h_dim=hdim,
            x_std_per_dim=noise_std_array,
            denoise_uses_y=denoise_uses_y
        ).to(self.device)

    def forward(self, x_context, y_context, x_target, y_target, toy_attention=True):
        p_y_pred, q_z_tgt, q_z_ctx, q_xs_ctx = self.dnp(
            x_context, y_context, x_target, y_target
        )

        nll = -p_y_pred.log_prob(y_target).sum(-1).mean()
        kl_z = kl_divergence(q_z_tgt, q_z_ctx).mean(dim=0).sum()
        mse  = ((p_y_pred.mean - y_target) ** 2).mean()

        kl_x = torch.tensor(0.0, device=x_context.device)
        for dim in self.denoise_dims:
            q_mean = q_xs_ctx.mean[:, :, dim:dim+1]
            q_std  = q_xs_ctx.stddev[:, :, dim:dim+1]
            q_dim  = Normal(q_mean, q_std)
            p_dim  = Normal(x_context[:, :, dim:dim+1],
                            self.noise_std_per_dim_dict[dim])
            kl_x   = kl_x + kl_divergence(q_dim, p_dim).mean(dim=0).sum()

        return p_y_pred, kl_z, nll, mse, kl_x


# ─────────────────────────────────────────────────────────────────────────────
# MEAN PREDICTOR BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def compute_mean_baseline(data_dir, feat_cfg, batch_size, c_set, test_set):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, train_stats = make_loader(data_dir, 'train', feat_cfg, batch_size)
    y_sum, y_count = 0.0, 0
    for _, y in train_loader:
        y_sum   += y.sum().item()
        y_count += y.numel()
    train_mean = y_sum / y_count

    test_loader, _ = make_loader(data_dir, 'test', feat_cfg, batch_size,
                                 train_norm_stats=train_stats)
    sq_errors, n_total = 0.0, 0
    for x, y in test_loader:
        _, _, x_t, y_t = chosen_context_target_split(x, y, c_set, test_set)
        y_t = y_t.to(device)
        sq_errors += ((y_t - train_mean) ** 2).sum().item()
        n_total   += y_t.numel()

    rmse = np.sqrt(sq_errors / n_total)
    return {'rmse': rmse, 'train_mean': train_mean}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _get_run_tag(feat_name, model_name):
    return f"{feat_name}__{model_name}"


def train_vanilla_np(data_dir, feat_cfg, feat_name, c_set, t_set,
                     save_dir, config):
    tag       = _get_run_tag(feat_name, 'vanilla_np')
    save_file = Path(save_dir) / f'{tag}.pth'

    if save_file.exists():
        print(f"  ✓ {tag}: already trained, loading checkpoint.")
        return torch.load(save_file, weights_only=False)

    print(f"\n  Training Vanilla NP | features={feat_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, train_stats = make_loader(
        data_dir, 'train', feat_cfg, config['batch_size'])
    val_loader, _ = make_loader(
        data_dir, 'val', feat_cfg, config['batch_size'],
        train_norm_stats=train_stats)

    model = NPregression(
        'NP', feat_cfg['xdim'], dim_output=1, dim_hidden=config['hdim']
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])

    history, best_val = {'train_loss': [], 'val_loss': []}, float('inf')

    for epoch in range(config['epochs']):
        model.train()
        t_losses = []
        for x, y in train_loader:
            x_c, y_c, x_t, y_t = chosen_context_target_split(x, y, c_set, t_set)
            x_c, y_c, x_t, y_t = [t.to(device) for t in (x_c, y_c, x_t, y_t)]
            _, kl, nll, _ = model(x_c, y_c, x_t, y_t)
            loss = nll + kl
            opt.zero_grad(); loss.backward(); opt.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x_c, y_c, x_t, y_t = chosen_context_target_split(x, y, c_set, t_set)
                x_c, y_c, x_t, y_t = [t.to(device) for t in (x_c, y_c, x_t, y_t)]
                _, kl, nll, _ = model(x_c, y_c, x_t, y_t)
                v_losses.append((nll + kl).item())

        avg_t, avg_v = np.mean(t_losses), np.mean(v_losses)
        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)
        if epoch % 5 == 0:
            print(f"    Ep {epoch:3d}: Train={avg_t:.4f} Val={avg_v:.4f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save({'model_state_dict': model.state_dict(),
                        'history': history, 'val_loss': best_val,
                        'config': config, 'feat_cfg': feat_cfg,
                        'norm_stats': train_stats}, save_file)

    print(f"    Best Val={best_val:.4f} | Saved → {save_file}")
    return torch.load(save_file, weights_only=False)


def train_dnp(data_dir, feat_cfg, feat_name, c_set, t_set,
              save_dir, config):
    tag       = _get_run_tag(feat_name, 'dnp')
    save_file = Path(save_dir) / f'{tag}.pth'

    if save_file.exists():
        print(f"  ✓ {tag}: already trained, loading checkpoint.")
        return torch.load(save_file, weights_only=False)

    print(f"\n  Training DNP | features={feat_name} | "
          f"denoise_dims={feat_cfg['denoise_dims']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, train_stats = make_loader(
        data_dir, 'train', feat_cfg, config['batch_size'])
    val_loader, _ = make_loader(
        data_dir, 'val', feat_cfg, config['batch_size'],
        train_norm_stats=train_stats)

    model = DenoisingNP(
        xdim=feat_cfg['xdim'], ydim=1, hdim=config['hdim'],
        noise_std_per_dim=feat_cfg['noise_std_per_dim'],
        denoise_dims=feat_cfg['denoise_dims']
    ).to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=config['lr'])
    beta = config['beta']

    history, best_val = {'train_loss': [], 'val_loss': []}, float('inf')

    for epoch in range(config['epochs']):
        model.train()
        t_losses = []
        for x, y in train_loader:
            x_c, y_c, x_t, y_t = chosen_context_target_split(x, y, c_set, t_set)
            x_c, y_c, x_t, y_t = [t.to(device) for t in (x_c, y_c, x_t, y_t)]
            _, kl_z, nll, _, kl_x = model(x_c, y_c, x_t, y_t)
            loss = nll + kl_z + beta * kl_x
            opt.zero_grad(); loss.backward(); opt.step()
            t_losses.append(loss.item())

        model.eval()
        v_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x_c, y_c, x_t, y_t = chosen_context_target_split(x, y, c_set, t_set)
                x_c, y_c, x_t, y_t = [t.to(device) for t in (x_c, y_c, x_t, y_t)]
                _, kl_z, nll, _, kl_x = model(x_c, y_c, x_t, y_t)
                v_losses.append((nll + kl_z + beta * kl_x).item())

        avg_t, avg_v = np.mean(t_losses), np.mean(v_losses)
        history['train_loss'].append(avg_t)
        history['val_loss'].append(avg_v)
        if epoch % 5 == 0:
            print(f"    Ep {epoch:3d}: Train={avg_t:.4f} Val={avg_v:.4f} β={beta:.3f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save({'model_state_dict': model.state_dict(),
                        'history': history, 'val_loss': best_val,
                        'config': config, 'feat_cfg': feat_cfg,
                        'norm_stats': train_stats}, save_file)

    print(f"    Best Val={best_val:.4f} | Saved → {save_file}")
    return torch.load(save_file, weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, test_loader, c_set, test_set, device, is_dnp=False, K=32):
    """MC-marginalized posterior-predictive metrics. z ~ q(z|context), K samples.
    RMSE from averaged predictive mean; NLL marginalized in density space via logsumexp.
    NP and DnNP evaluated identically (no y_target leak)."""
    model.eval()
    total_nll, total_se, n_pts = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x_c, y_c, x_t, y_t = chosen_context_target_split(x, y, c_set, test_set)
            x_c, y_c, x_t, y_t = [t.to(device) for t in (x_c, y_c, x_t, y_t)]

            logps, mus = [], []
            for _ in range(K):
                p = model.dnp(x_c, y_c, x_t) if is_dnp else model(x_c, y_c, x_t, None)
                logps.append(p.log_prob(y_t).sum(-1))   # (B, T)
                mus.append(p.mean)                       # (B, T, ydim)

            logps = torch.stack(logps, dim=0)            # (K, B, T)
            mus   = torch.stack(mus, dim=0)              # (K, B, T, ydim)

            nll_pt = -(torch.logsumexp(logps, dim=0) - np.log(K))   # (B, T)
            se = ((mus.mean(dim=0) - y_t) ** 2).sum(-1)             # (B, T)

            total_nll += nll_pt.sum().item()
            total_se  += se.sum().item()
            n_pts     += nll_pt.numel()

    return {'nll': total_nll / n_pts, 'rmse': float(np.sqrt(total_se / n_pts))}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    set_seed(BASE_CONFIG['seed'])
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(BASE_CONFIG['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = BASE_CONFIG['data_dir']
    set_path = Path(data_dir) / 'context_target_test.txt'

    splits = load_splits(set_path)
    c_set_str = splits['context_set']
    t_set_str = splits['target_set']
    test_set_str = splits['test_set']

    # Convert string station IDs to integer positions using the dataset's ordering.
    # Build the mapping once from a probe dataset; it's stable across splits.
    probe_ds = NoaaTemperatureDataset(data_dir, split='train')
    sid2idx = probe_ds.station_to_idx

    def to_indices(string_ids):
        return [sid2idx[s] for s in string_ids if s in sid2idx]

    c_set = to_indices(c_set_str)
    t_set = to_indices(t_set_str)
    test_set = to_indices(test_set_str)

    print(f"Splits (int indices): context={len(c_set)} target={len(t_set)} "
        f"held-out={len(test_set)}")
    print(f"Total stations in dataset: {len(sid2idx)}")
    if len(c_set) + len(t_set) + len(test_set) != len(sid2idx):
        print(f"⚠ WARNING: split sizes don't sum to total stations — some "
            f"split stations were dropped by completeness filter")
    all_results = {}

    for feat_name, feat_cfg in FEATURE_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"FEATURE SET: {feat_name}  (xdim={feat_cfg['xdim']}, "
              f"denoise_dims={feat_cfg['denoise_dims']})")
        print('='*70)

        results = {}

        # ── 1. Mean predictor baseline ─────────────────────────────────────
        print("\n[1/3] Mean predictor baseline...")
        mean_res = compute_mean_baseline(
            data_dir, feat_cfg, BASE_CONFIG['batch_size'], c_set, test_set
        )
        results['mean_baseline'] = mean_res
        print(f"      Train mean={mean_res['train_mean']:.4f} | "
              f"Test RMSE={mean_res['rmse']:.4f}")

        # ── 2. Vanilla NP ──────────────────────────────────────────────────
        print("\n[2/3] Vanilla NP...")
        train_vanilla_np(data_dir, feat_cfg, feat_name, c_set, t_set,
                         save_dir, BASE_CONFIG)

        vnp_ckpt = torch.load(save_dir / f'{feat_name}__vanilla_np.pth',
                              weights_only=False)
        vnp_model = NPregression(
            'NP', feat_cfg['xdim'], dim_output=1, dim_hidden=BASE_CONFIG['hdim']
        ).to(device)
        vnp_model.load_state_dict(vnp_ckpt['model_state_dict'])

        test_loader, _ = make_loader(
            data_dir, 'test', feat_cfg, BASE_CONFIG['batch_size'],
            train_norm_stats=vnp_ckpt['norm_stats'])
        results['vanilla_np'] = evaluate_model(
            vnp_model, test_loader, c_set, test_set, device, is_dnp=False
        )

        # ── 3. DNP ─────────────────────────────────────────────────────────
        print("\n[3/3] DNP...")
        train_dnp(data_dir, feat_cfg, feat_name, c_set, t_set,
                  save_dir, BASE_CONFIG)

        dnp_ckpt = torch.load(save_dir / f'{feat_name}__dnp.pth',
                              weights_only=False)
        dnp_model = DenoisingNP(
            xdim=feat_cfg['xdim'], ydim=1, hdim=BASE_CONFIG['hdim'],
            noise_std_per_dim=feat_cfg['noise_std_per_dim'],
            denoise_dims=feat_cfg['denoise_dims']
        ).to(device)
        dnp_model.load_state_dict(dnp_ckpt['model_state_dict'])

        test_loader, _ = make_loader(
            data_dir, 'test', feat_cfg, BASE_CONFIG['batch_size'],
            train_norm_stats=dnp_ckpt['norm_stats'])
        results['dnp'] = evaluate_model(
            dnp_model, test_loader, c_set, test_set, device, is_dnp=True
        )

        all_results[feat_name] = results

    # ── Final table ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*90}")
    header = (f"{'Feature set':<20} {'Mean RMSE':>10} "
              f"{'VNP NLL':>10} {'VNP RMSE':>10} "
              f"{'DNP NLL':>10} {'DNP RMSE':>10}")
    print(header)
    print('-'*90)

    for feat_name, res in all_results.items():
        m, v, d = res['mean_baseline'], res['vanilla_np'], res['dnp']
        row = (f"{feat_name:<20} {m['rmse']:>10.4f} "
               f"{v['nll']:>10.4f} {v['rmse']:>10.4f} "
               f"{d['nll']:>10.4f} {d['rmse']:>10.4f}")
        print(row)
    print('='*90)

    summary_file = save_dir / 'full_comparison.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results → {summary_file}")


if __name__ == '__main__':
    main()
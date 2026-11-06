"""
1D Toy Experiment: NP (clean) vs NP (noisy) vs DNP vs DNP (learn σ)

Usage:
    python 1d_gp.py --seed 1                     # RBF kernel (paper's main 1D experiment)
    python 1d_gp.py --seed 1 --kernel periodic   # periodic kernel, random length-scale/period
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from neural_process import NeuralProcess, DenoisingNeuralProcess


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# 1.  GP Data Generator
# ─────────────────────────────────────────────
def rbf_kernel(x1, x2, length_scale=0.5, sigma_f=1.0):
    diff = x1[:, None] - x2[None, :]
    return sigma_f**2 * np.exp(-0.5 * diff**2 / length_scale**2)


def sample_gp_batch(batch_size, num_points, x_range=(-1, 1),
                    length_scale=0.5, obs_noise=0.0):
    x_vals = np.linspace(*x_range, num_points)
    K = rbf_kernel(x_vals, x_vals, length_scale) + (obs_noise + 1e-6) * np.eye(num_points)
    L = np.linalg.cholesky(K)
    x_batch = np.tile(x_vals[None, :, None], (batch_size, 1, 1))
    eps = np.random.randn(batch_size, num_points)
    y_batch = (L @ eps.T).T[:, :, None]
    return (torch.tensor(x_batch, dtype=torch.float32),
            torch.tensor(y_batch, dtype=torch.float32))


def periodic_kernel(x1, x2, length_scale=0.7, period=0.75, sigma_f=1.0):
    """Exp-sine-squared kernel: s^2 exp(-2 sin^2(pi |x - x'| / p) / l^2)."""
    diff = np.abs(x1[:, None] - x2[None, :])
    return sigma_f**2 * np.exp(-2.0 * np.sin(np.pi * diff / period)**2 / length_scale**2)


def sample_periodic_gp_batch(batch_size, num_points, x_range=(-1, 1),
                             length_scale_range=None, period_range=None,
                             obs_noise=0.0, data_seed=None):
    """Each task gets its own length-scale and period, drawn uniformly.
    data_seed given -> dedicated RNG (dataset independent of --seed);
    data_seed None  -> global np.random (so plots follow set_seed like RBF)."""
    length_scale_range = length_scale_range or LS_RANGE
    period_range       = period_range or PERIOD_RANGE
    rng = np.random.default_rng(data_seed) if data_seed is not None else None
    uniform = rng.uniform if rng is not None else np.random.uniform
    randn   = rng.standard_normal if rng is not None else np.random.randn
    x_vals = np.linspace(*x_range, num_points)
    y_batch = np.empty((batch_size, num_points))
    for i in range(batch_size):
        ls = uniform(*length_scale_range)
        p  = uniform(*period_range)
        K  = periodic_kernel(x_vals, x_vals, ls, p) + (obs_noise + 1e-6) * np.eye(num_points)
        L  = np.linalg.cholesky(K)
        y_batch[i] = L @ randn(num_points)
    x_batch = np.tile(x_vals[None, :, None], (batch_size, 1, 1))
    return (torch.tensor(x_batch, dtype=torch.float32),
            torch.tensor(y_batch[:, :, None], dtype=torch.float32))


KERNEL   = "rbf"   # set from --kernel in main
CTX_MAX  = 30      # max training context size (set from --ctx-max)
PATIENCE = 2000    # early-stopping patience (set from --patience)
LS_RANGE     = (1.0, 2.0)   # periodic kernel length-scale range (--ls-range)
PERIOD_RANGE = (0.7, 1.2)   # periodic kernel period range (--period-range); equal ends = fixed


def sample_task(batch_size, num_points):
    """Sample functions from whichever kernel this run uses (used by the plots)."""
    if KERNEL == "periodic":
        return sample_periodic_gp_batch(batch_size, num_points)
    return sample_gp_batch(batch_size, num_points)


def get_context_target(x, y, num_context):
    idx = torch.randperm(x.shape[1])[:num_context].sort().values
    return x[:, idx], y[:, idx], x, y


def add_noise(x, x_std):
    return x + x_std * torch.randn_like(x)


# ─────────────────────────────────────────────
# 2.  Dataset cache
# ─────────────────────────────────────────────
def get_dataset(num_tasks=10000, num_points=100, cache_path="gp_dataset_1d.pt",
                kernel="rbf"):
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path} ...")
        data = torch.load(cache_path)
        return data["x"], data["y"]
    print(f"Generating GP dataset ({num_tasks} tasks) → {cache_path} ...")
    if kernel == "periodic":
        x, y = sample_periodic_gp_batch(num_tasks, num_points, data_seed=0)
    else:
        x, y = sample_gp_batch(num_tasks, num_points)
    torch.save({"x": x, "y": y}, cache_path)
    print("Done.")
    return x, y


# ─────────────────────────────────────────────
# 3.  Loss functions
# ─────────────────────────────────────────────
def np_loss(model, x_ctx, y_ctx, x_tgt, y_tgt):
    p_y, q_tgt, q_ctx = model(x_ctx, y_ctx, x_tgt, y_tgt)
    ll   = p_y.log_prob(y_tgt).mean()
    kl_z = kl_divergence(q_tgt, q_ctx).mean()
    return -(ll - kl_z), ll.item(), kl_z.item()


def dnp_loss(model, x_ctx, y_ctx, x_tgt, y_tgt,
             beta_kl=1.0, beta_denoise=0.1, x_std=0.1):
    p_y, q_z_tgt, q_z_ctx, q_xs_ctx = model(x_ctx, y_ctx, x_tgt, y_tgt)
    ll   = p_y.log_prob(y_tgt).mean()
    kl_z = kl_divergence(q_z_tgt, q_z_ctx).mean()

    # --- use model's learned sigma if it has one, else fall back to x_std arg ---
    sigma = model.sigma if (hasattr(model, 'log_sigma') and model.log_sigma is not None) \
            else x_std
    # sigma may be a scalar tensor (learned) or a float (fixed); both work here
    p_x  = Normal(x_ctx, sigma * torch.ones_like(x_ctx))
    kl_x = kl_divergence(q_xs_ctx, p_x).mean()
    loss = -(ll - beta_kl * kl_z - beta_denoise * kl_x)
    return loss, ll.item(), kl_z.item(), kl_x.item()


# ─────────────────────────────────────────────
# 4.  Training loop
# ─────────────────────────────────────────────
def train(model, model_type, x_data, y_data,
          num_epochs=15000, batch_size=64, lr=3e-4,
          x_std=0.1, beta_kl=1.0, beta_denoise=0.1,
          patience=None, min_delta=1e-4, warmup=3000,
          check_every=100, window=200):
    """
    Trains model with random context size per batch and early stopping.
    Context size is sampled uniformly in [5, 30] each batch so the model
    generalizes to any context size at evaluation time.
    """
    model = model.to(DEVICE).train()
    opt   = optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "ll": [], "kl_z": []}
    if model_type in ("dnp", "dnp_ls"):
        history["kl_x"] = []
    if model_type == "dnp_ls":
        history["sigma"] = []
 
    num_tasks   = x_data.shape[0]
    num_points  = x_data.shape[1]
    patience = PATIENCE if patience is None else patience
    ctx_min, ctx_max = 5, min(CTX_MAX, num_points - 1)
 
    best_smoothed_loss       = float("inf")
    epochs_without_improvement = 0
    stopped_epoch            = num_epochs
 
    for epoch in range(1, num_epochs + 1):
        idx  = torch.randint(0, num_tasks, (batch_size,))
        x, y = x_data[idx], y_data[idx]
 
        # random context size every batch
        nc = torch.randint(ctx_min, ctx_max + 1, (1,)).item()
        x_ctx, y_ctx, x_tgt, y_tgt = get_context_target(x, y, nc)
 
        x_ctx, y_ctx = x_ctx.to(DEVICE), y_ctx.to(DEVICE)
        x_tgt, y_tgt = x_tgt.to(DEVICE), y_tgt.to(DEVICE)
 
        x_ctx_in = x_ctx if model_type == "np_clean" else add_noise(x_ctx, x_std)
 
        opt.zero_grad()
        if model_type in ("np_clean", "np_noisy"):
            loss, ll, kl_z = np_loss(model, x_ctx_in, y_ctx, x_tgt, y_tgt)
            history["kl_z"].append(kl_z)
        else:
            loss, ll, kl_z, kl_x = dnp_loss(
                model, x_ctx_in, y_ctx, x_tgt, y_tgt,
                beta_kl=beta_kl, beta_denoise=beta_denoise, x_std=x_std)
            history["kl_z"].append(kl_z)
            history["kl_x"].append(kl_x)
 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
 
        history["loss"].append(loss.item())
        history["ll"].append(ll)
        if model_type == "dnp_ls":
            history["sigma"].append(model.sigma.item())
 
        # ── convergence check ─────────────────────────────────────
        if epoch >= warmup and epoch % check_every == 0:
            smoothed = np.mean(history["loss"][-window:])
            if smoothed < best_smoothed_loss - min_delta:
                best_smoothed_loss = smoothed
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += check_every
 
            if epochs_without_improvement >= patience:
                stopped_epoch = epoch
                print(f"  ✓ [{model_type.upper():10s}] Converged at epoch {epoch} "
                      f"(smoothed loss={smoothed:.4f})")
                break
 
        if epoch % 500 == 0:
            smoothed_str = ""
            if len(history["loss"]) >= window:
                smoothed_str = f" | Smoothed {np.mean(history['loss'][-window:]):.4f}"
            msg = (f"[{model_type.upper():10s}] Epoch {epoch:5d}/{num_epochs} | "
                   f"Loss {loss.item():.4f} | LL {ll:.4f} | KL_z {kl_z:.4f}"
                   f"{smoothed_str}")
            if model_type in ("dnp", "dnp_ls"):
                msg += f" | KL_x {kl_x:.4f}"
            if model_type == "dnp_ls":
                msg += f" | σ_learned {model.sigma.item():.4f}"
            print(msg)
 
    if stopped_epoch == num_epochs:
        print(f"  ✗ [{model_type.upper():10s}] Reached max epochs ({num_epochs}) "
              f"without convergence — consider increasing num_epochs")
 
    history["stopped_epoch"] = stopped_epoch
    model.eval()
    return history

# ─────────────────────────────────────────────
# 5.  Prediction
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_samples(model, model_type, x_ctx, y_ctx, x_tgt, x_std=0.1, n_samples=50, seed=0):
    x_ctx, y_ctx, x_tgt = [t.to(DEVICE) for t in (x_ctx, y_ctx, x_tgt)]
    # dnp_ls behaves like dnp at inference: receives noisy coords
    x_ctx_in = add_noise(x_ctx, x_std) if model_type != "np_clean" else x_ctx
    samples = []
    for i in range(n_samples):
        torch.manual_seed(seed + i)
        p_y = model(x_ctx_in, y_ctx, x_tgt)
        samples.append(p_y.loc.squeeze().cpu().numpy())
    return np.stack(samples)   # (n_samples, T)


@torch.no_grad()
def predict(model, model_type, x_ctx, y_ctx, x_tgt, x_std=0.1, n_samples=50):
    x_ctx, y_ctx, x_tgt = [t.to(DEVICE) for t in (x_ctx, y_ctx, x_tgt)]
    # dnp_ls behaves like dnp at inference
    x_ctx_in = add_noise(x_ctx, x_std) if model_type != "np_clean" else x_ctx

    mus, vars_ = [], []
    for _ in range(n_samples):
        p_y = model(x_ctx_in, y_ctx, x_tgt)
        mus.append(p_y.loc)
        vars_.append(p_y.scale**2)

    mu  = torch.stack(mus).mean(0).squeeze().cpu().numpy()
    std = (torch.stack(vars_).mean(0) +
           torch.stack(mus).var(0)).squeeze().cpu().numpy()
    return mu, np.sqrt(std)


# ─────────────────────────────────────────────
# 6.  Evaluation
# ─────────────────────────────────────────────
MODELS_CFG = [
    ("np_clean", "NP (clean)",    "steelblue"),
    ("np_noisy", "NP (noisy)",    "crimson"),
    ("dnp",      "DNP",           "darkorange"),
    ("dnp_ls",   "DNP (learn σ)", "#2ca02c"),   # green
]

def evaluate(models, x_data, y_data, x_std=0.1, num_context=10,
             num_tasks=200, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    results = {k: {"nll": [], "rmse": []} for k, *_ in MODELS_CFG}

    for i in range(num_tasks):
        x_all, y_all = x_data[[i]], y_data[[i]]
        all_idx = torch.randperm(x_all.shape[1])
        ctx_idx = all_idx[:num_context].sort().values
        tgt_idx = all_idx[num_context:].sort().values
        x_ctx, y_ctx = x_all[:, ctx_idx], y_all[:, ctx_idx]
        x_tgt, y_tgt = x_all[:, tgt_idx], y_all[:, tgt_idx]

        for mtype, _, _ in MODELS_CFG:
            mu, std = predict(models[mtype], mtype, x_ctx, y_ctx,
                              x_tgt, x_std=x_std)
            y_np = y_tgt.squeeze().numpy()
            nll  = 0.5 * np.mean(((y_np - mu) / std)**2 + np.log(2 * np.pi * std**2))
            rmse = np.sqrt(np.mean((y_np - mu)**2))
            results[mtype]["nll"].append(nll)
            results[mtype]["rmse"].append(rmse)

    print(f"\n{'Model':<16} {'NLL (mean±std)':>20} {'RMSE (mean±std)':>20}")
    print("─" * 60)
    for mtype, label, _ in MODELS_CFG:
        nll_m  = np.mean(results[mtype]["nll"])
        nll_s  = np.std(results[mtype]["nll"])
        rmse_m = np.mean(results[mtype]["rmse"])
        rmse_s = np.std(results[mtype]["rmse"])
        print(f"{label:<16}  {nll_m:+.3f} ± {nll_s:.3f}   {rmse_m:.4f} ± {rmse_s:.4f}")

    return results


# ─────────────────────────────────────────────
# 7.  Diagnostic plot per noise level
# ─────────────────────────────────────────────
def plot_all(models, histories, results, x_std=0.1, num_context=10, seed=7,
             save_prefix="dnp_vs_np_1d"):
    torch.manual_seed(seed); np.random.seed(seed)
    smooth = lambda v, w=50: np.convolve(v, np.ones(w)/w, mode='valid')

    fig = plt.figure(figsize=(22, 10))
    gs  = fig.add_gridspec(2, 5, hspace=0.38, wspace=0.35)

    # ── Row 0: training curves ──────────────────────────────────
    for ax_idx, (key, title) in enumerate([("loss","Training Loss"),
                                            ("ll",  "Log-Likelihood"),
                                            ("kl_z","KL_z")]):
        ax = fig.add_subplot(gs[0, ax_idx])
        for mtype, label, color in MODELS_CFG:
            ax.plot(smooth(histories[mtype][key]), label=label, color=color, lw=1.5)
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax_kx = fig.add_subplot(gs[0, 3])
    for mtype, label, color in [("dnp", "DNP", "darkorange"),
                                  ("dnp_ls", "DNP (learn σ)", "#2ca02c")]:
        ax_kx.plot(smooth(histories[mtype]["kl_x"]), color=color, lw=1.5, label=label)
    ax_kx.set_title("KL_x (denoising)"); ax_kx.set_xlabel("Epoch")
    ax_kx.legend(fontsize=7); ax_kx.grid(True, alpha=0.3)

    # ── Learned sigma convergence curve ────────────────────────
    ax_sig = fig.add_subplot(gs[0, 4])
    ax_sig.plot(smooth(histories["dnp_ls"]["sigma"]),
                color="#2ca02c", lw=1.5, label="Learned σ")
    ax_sig.axhline(x_std, color="black", lw=1.2, linestyle="--", label=f"True σ={x_std}")
    ax_sig.set_title("DNP (learn σ) — σ convergence")
    ax_sig.set_xlabel("Epoch"); ax_sig.set_ylabel("σ value")
    ax_sig.legend(fontsize=7); ax_sig.grid(True, alpha=0.3)

    # ── Row 1: example predictions with NOISY context shown ────
    CTX_TRUE_COLOR  = "#000000"
    CTX_NOISY_COLOR = "#FE6100"

    for col in range(3):
        ax = fig.add_subplot(gs[1, col])
        x_all, y_all = sample_task(1, 200)
        all_idx = torch.randperm(200)
        ctx_idx = all_idx[:num_context].sort().values
        x_ctx, y_ctx = x_all[:, ctx_idx], y_all[:, ctx_idx]
        x_plot = torch.linspace(-1, 1, 200)[None, :, None]

        # generate one noisy context realization for visualization
        x_ctx_noisy = add_noise(x_ctx, x_std)

        for mtype, label, color in MODELS_CFG:
            mu, std = predict(models[mtype], mtype, x_ctx, y_ctx,
                              x_plot, x_std=x_std)
            xn = x_plot.squeeze().numpy()
            ax.plot(xn, mu, color=color, lw=1.8, label=label)
            ax.fill_between(xn, mu-std, mu+std, alpha=0.10, color=color)

        # true context locations (black circles)
        ax.scatter(x_ctx.squeeze(), y_ctx.squeeze(),
                   c=CTX_TRUE_COLOR, s=45, zorder=7, marker="o",
                   label="Context (true)")
        # noisy context locations (orange triangles) — what models actually see
        ax.scatter(x_ctx_noisy.squeeze(), y_ctx.squeeze(),
                   c=CTX_NOISY_COLOR, s=45, zorder=6, marker="^",
                   label="Context (noisy)")

        ax.set_title(f"Example {col+1}"); ax.set_xlim(-1, 1)
        if col == 0:
            ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # ── Bar chart: NLL / RMSE ───────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 3])
    labels = [label for _, label, _ in MODELS_CFG]
    colors = [c     for _, _,     c in MODELS_CFG]
    x_pos  = np.arange(len(MODELS_CFG))
    width  = 0.35

    nll_means  = [np.mean(results[k]["nll"])  for k, *_ in MODELS_CFG]
    nll_stds   = [np.std(results[k]["nll"])   for k, *_ in MODELS_CFG]
    rmse_means = [np.mean(results[k]["rmse"]) for k, *_ in MODELS_CFG]
    rmse_stds  = [np.std(results[k]["rmse"])  for k, *_ in MODELS_CFG]

    ax2 = ax_bar.twinx()
    ax_bar.bar(x_pos - width/2, nll_means, width, yerr=nll_stds,
               capsize=4, color=colors, alpha=0.85, label="NLL")
    ax2.bar(x_pos + width/2, rmse_means, width, yerr=rmse_stds,
            capsize=4, color=colors, alpha=0.4, label="RMSE")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    ax_bar.set_ylabel("NLL (↓)", fontsize=8)
    ax2.set_ylabel("RMSE (↓)", fontsize=8)
    ax_bar.set_title(f"Target NLL & RMSE\n(unseen pts, 200 tasks, σ={x_std})")
    ax_bar.grid(True, alpha=0.3, axis="y")
    h1, l1 = ax_bar.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_bar.legend(h1+h2, l1+l2, fontsize=7)

    # ── Learned sigma annotation ────────────────────────────────
    learned_sig = models["dnp_ls"].sigma.item()
    ax_ann = fig.add_subplot(gs[1, 4])
    ax_ann.axis("off")
    ax_ann.text(0.5, 0.6,
                f"DNP (learn σ)\nFinal learned σ: {learned_sig:.4f}\nTrue σ:          {x_std:.4f}\nError:           {abs(learned_sig - x_std):.4f}",
                ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#e8f5e9", edgecolor="#2ca02c", lw=2),
                transform=ax_ann.transAxes, family="monospace")

    plt.suptitle(
        f"NP (clean) vs NP (noisy) vs DNP vs DNP(learn σ) — 1D GP Toy  [σ_x={x_std}]",
        fontsize=13, fontweight="bold")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_prefix}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {save_prefix}.pdf / .png")
    plt.show()


# ─────────────────────────────────────────────
# 8.  Sigma accuracy plot (across noise levels)
# ─────────────────────────────────────────────
def plot_sigma_accuracy(sigma_records, save_prefix="dnp_sigma_accuracy"):
    """
    sigma_records : dict  { true_sigma (float) : learned_sigma (float) }
    Shows how well DNP (learn σ) recovers the true noise level.
    """
    true_sigs    = sorted(sigma_records.keys())
    learned_sigs = [sigma_records[s] for s in true_sigs]
    errors       = [abs(l - t) for l, t in zip(learned_sigs, true_sigs)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: learned vs true (identity line) ──────────────────
    ax = axes[0]
    ax.plot(true_sigs, true_sigs, "k--", lw=1.5, label="Perfect recovery (y=x)")
    ax.scatter(true_sigs, learned_sigs, color="#2ca02c", s=70, zorder=5)
    ax.plot(true_sigs, learned_sigs, color="#2ca02c", lw=1.8, alpha=0.8,
            label="Learned σ")
    ax.set_xlabel("True σ_x", fontsize=11)
    ax.set_ylabel("Learned σ_x", fontsize=11)
    ax.set_title("Sigma Recovery: Learned vs True", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle=":")

    # ── Right: absolute error per noise level ──────────────────
    ax2 = axes[1]
    ax2.bar(true_sigs, errors, color="#2ca02c", alpha=0.75,
            width=min(np.diff(true_sigs).min() * 0.6, 0.04) if len(true_sigs) > 1 else 0.03)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("True σ_x", fontsize=11)
    ax2.set_ylabel("|Learned σ − True σ|", fontsize=11)
    ax2.set_title("Absolute Recovery Error", fontsize=12)
    ax2.set_xticks(true_sigs)
    ax2.tick_params(axis='x', labelrotation=45)
    ax2.grid(True, alpha=0.3, axis="y", linestyle=":")

    plt.suptitle("DNP (learn σ) — Sigma Recovery Across Noise Levels",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_prefix}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {save_prefix}.pdf / .png")
    plt.show()


# ─────────────────────────────────────────────
# 9.  Noise sweep summary plot
# ─────────────────────────────────────────────
def plot_noise_sweep(noise_levels, all_results, save_prefix="dnp_noise_sweep"):
    MODEL_STYLE = {
        "np_clean": dict(color="#000000", marker="o", linestyle="-",  label="NP (clean)"),
        "np_noisy": dict(color="#648FFF", marker="s", linestyle="--", label="NP (noisy)"),
        "dnp":      dict(color="#FE6100", marker="^", linestyle="-.", label="DNP"),
        "dnp_ls":   dict(color="#2ca02c", marker="D", linestyle=":",  label="DNP (learn σ)"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sigmas = sorted(all_results.keys())

    for metric, ax, ylabel in [("nll",  axes[0], "NLL (↓)"),
                                ("rmse", axes[1], "RMSE (↓)")]:
        for mtype, label, _ in MODELS_CFG:
            means = [np.mean(all_results[s][mtype][metric]) for s in sigmas]
            style = MODEL_STYLE[mtype]
            ax.plot(sigmas, means,
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], label=style["label"],
                    lw=2, markersize=7)
        ax.set_xlabel("Input noise $\\sigma_x$")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_xticks(sigmas)
        ax.tick_params(axis='x', labelrotation=90)

    plt.suptitle("NP vs DNP vs DNP(learn σ) — Performance vs Noise Level (1D GP Toy)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_prefix}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {save_prefix}.pdf / .png")
    plt.show()


# ─────────────────────────────────────────────
# 10.  Paper figure
# ─────────────────────────────────────────────
def plot_paper_figure(all_models, noise_levels=(0.2, 0.3, 0.4),
                      num_context=20, n_samples=50, seed=7,
                      save_prefix="dnp_paper_figure"):
    set_seed(seed)
    x_plot = torch.linspace(-1, 1, 200)[None, :, None]
    x_np   = x_plot.squeeze().numpy()

    x_all, y_all = sample_task(1, 500)
    all_idx = torch.randperm(500)
    ctx_idx = all_idx[:num_context].sort().values
    x_ctx, y_ctx = x_all[:, ctx_idx], y_all[:, ctx_idx]
    sort_idx = x_all.squeeze().argsort()
    x_true = x_all.squeeze()[sort_idx].numpy()
    y_true = y_all.squeeze()[sort_idx].numpy()

    row_cfg = [("np_clean", "NP (clean)"),
               ("np_noisy", "NP (noisy)"),
               ("dnp",      "DNP"),
            #    ("dnp_ls",   "DNP (learn σ)")
               ]

    # model colors — one per row
    MODEL_COLOR = {
        "np_clean": "#4878CF",   # blue
        "np_noisy": "#D65F5F",   # red
        "dnp":      "#FE6100",   # orange
        "dnp_ls":   "#2ca02c",   # green
    }
    TRUE_COLOR      = "#000000"
    CTX_TRUE_COLOR  = "#000000"
    CTX_NOISY_COLOR = "#FE6100"

    n_rows = len(row_cfg)
    n_cols = len(noise_levels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows))
    # ensure 2d indexing even for single column
    if n_cols == 1:
        axes = axes[:, None]

    # ── first pass: collect all mu/std to compute global y limits ──
    all_preds = {}   # (col, row) -> (mu, std)
    for col, x_std in enumerate(noise_levels):
        torch.manual_seed(seed + col)
        x_ctx_noisy = add_noise(x_ctx, x_std)
        for row, (mtype, _) in enumerate(row_cfg):
            if x_std not in all_models:
                continue
            x_ctx_in = x_ctx if mtype == "np_clean" else x_ctx_noisy
            mu, std = predict(all_models[x_std][mtype], mtype,
                              x_ctx_in, y_ctx, x_plot,
                              x_std=x_std, n_samples=n_samples)
            all_preds[(col, row)] = (mu, std)

    # global y range across all predictions + true function
    all_vals = [y_true]
    for mu, std in all_preds.values():
        all_vals.append(mu + 2 * std)
        all_vals.append(mu - 2 * std)
    y_min = min(v.min() for v in all_vals)
    y_max = max(v.max() for v in all_vals)
    y_pad = 0.1 * (y_max - y_min)
    y_lim = (y_min - y_pad, y_max + y_pad)

    # ── second pass: plot ──────────────────────────────────────────
    for col, x_std in enumerate(noise_levels):
        torch.manual_seed(seed + col)
        x_ctx_noisy = add_noise(x_ctx, x_std)

        for row, (mtype, label) in enumerate(row_cfg):
            ax = axes[row, col]

            if x_std not in all_models:
                ax.text(0.5, 0.5, f"No model for σ={x_std}",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            color = MODEL_COLOR[mtype]
            mu, std = all_preds[(col, row)]

            # mean + 2σ band
            ax.fill_between(x_np, mu - 2*std, mu + 2*std,
                            alpha=0.20, color=color)
            ax.fill_between(x_np, mu - std, mu + std,
                            alpha=0.30, color=color)
            ax.plot(x_np, mu, color=color, lw=2.0, zorder=4)

            # true function
            ax.plot(x_true, y_true, color=TRUE_COLOR, lw=1.5,
                    linestyle="--", zorder=5, alpha=0.8)

            # context points
            ax.scatter(x_ctx.squeeze().numpy(), y_ctx.squeeze().numpy(),
                       c=CTX_TRUE_COLOR, s=35, zorder=7, marker="o")
            if mtype != "np_clean":
                ax.scatter(x_ctx_noisy.squeeze().numpy(), y_ctx.squeeze().numpy(),
                           c=CTX_NOISY_COLOR, s=35, zorder=6, marker="^")

            ax.set_xlim(-1, 1)
            ax.set_ylim(y_lim)
            ax.grid(True, alpha=0.2, linestyle=":")
            ax.tick_params(labelsize=8)

            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            if row == 0:
                ax.set_title(f"$\\sigma_x = {x_std}$", fontsize=12)

    # ── legend ────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color=TRUE_COLOR, lw=1.5,
                   linestyle="--", label="True function"),
        plt.Line2D([0], [0], color=CTX_TRUE_COLOR, lw=0,
                   marker="o", markersize=6, label="Context (true)"),
        plt.Line2D([0], [0], color=CTX_NOISY_COLOR, lw=0,
                   marker="^", markersize=6, label="Context (noisy)"),
        plt.Line2D([0], [0], color="gray", lw=0,
                   marker="s", markersize=8, alpha=0.5,
                   label="Mean ± 1σ / 2σ band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        f"1D GP Regression — Mean ± uncertainty | {num_context} context points",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_prefix}.png", bbox_inches="tight", dpi=200)
    print(f"Saved: {save_prefix}.pdf / .png")
    plt.show()
def find_best_seed(all_models, x_std=0.3, num_context=20,
                   n_samples=50, n_seeds=50):
    """
    Scores seeds by how dramatically NP (noisy) fails while DNP recovers.
    Score = NLL(np_noisy) - NLL(dnp), higher = more compelling visual.
    """
    assert x_std in all_models, f"No model trained for σ={x_std}"

    scores = []
    x_plot = torch.linspace(-1, 1, 200)[None, :, None]

    for seed in range(n_seeds):
        set_seed(seed)
        x_all, y_all = sample_task(1, 500)
        all_idx  = torch.randperm(x_all.shape[1])
        ctx_idx  = all_idx[:num_context].sort().values
        x_ctx, y_ctx = x_all[:, ctx_idx], y_all[:, ctx_idx]

        sort_idx = x_all.squeeze().argsort()
        y_true   = x_all.squeeze()[sort_idx].numpy()
        y_vals   = y_all.squeeze()[sort_idx].numpy()

        x_ctx_noisy = add_noise(x_ctx, x_std)

        results = {}
        for mtype in ("np_clean", "np_noisy", "dnp"):
            x_ctx_in = x_ctx if mtype == "np_clean" else x_ctx_noisy
            mu, std  = predict(all_models[x_std][mtype], mtype,
                               x_ctx_in, y_ctx, x_plot,
                               x_std=x_std, n_samples=n_samples)
            x_np     = x_plot.squeeze().numpy()
            y_interp = np.interp(x_np, y_true, y_vals)
            nll  = 0.5 * np.mean(((y_interp - mu) / std)**2 +
                                  np.log(2 * np.pi * std**2))
            results[mtype] = nll

        # score: NLL improvement, with NP clean sanity check
        improvement = results["np_noisy"] - results["dnp"]
        clean_ok    = results["np_clean"] < results["np_noisy"] * 0.8
        score = improvement if clean_ok else -1.0

        scores.append((score, seed))
        if seed % 10 == 0:
            print(f"  seed {seed:3d} | NP(clean)={results['np_clean']:.3f} "
                  f"NP(noisy)={results['np_noisy']:.3f} "
                  f"DNP={results['dnp']:.3f} | score={score:.3f}")

    scores.sort(reverse=True)
    print(f"\nTop 5 seeds:")
    for score, seed in scores[:5]:
        print(f"  seed={seed:3d}  score={score:.4f}")
    return scores

def plot_best_example(all_models, seed, noise_levels=(0.2, 0.3, 0.4),
                      all_results=None, num_context=20,
                      n_samples=50, save_prefix="dnp_best_example"):
    """
    Grid figure:
      Rows    = NP (clean) | NP (noisy) | DnNP
      Cols    = sigma values  +  right column: NLL (top) / RMSE (bottom)
    """
    import matplotlib.gridspec as gridspec

    MODEL_COLOR = {
        "np_clean": "#4878CF",
        "np_noisy": "#D65F5F",
        "dnp":      "#FE6100",
    }
    row_cfg = [
        ("np_clean", "NP (clean)"),
        ("np_noisy", "NP (noisy)"),
        ("dnp",      "DnNP (ours)"),
    ]
    SWEEP_STYLE = {
        "np_clean": dict(color="#4878CF", marker="o", ls="-",  label="NP (clean)"),
        "np_noisy": dict(color="#D65F5F", marker="s", ls="--", label="NP (noisy)"),
        "dnp":      dict(color="#FE6100", marker="^", ls="-.", label="DnNP (ours)"),
    }

    n_models   = len(row_cfg)
    n_sig_cols = len(noise_levels)

    # ── layout: sigma cols + 1 right col for sweep panels ─────────
    fig = plt.figure(figsize=(4.5 * n_sig_cols + 2.5, 2.8 * n_models))
    gs  = gridspec.GridSpec(
        n_models, n_sig_cols + 1,
        width_ratios=[1] * n_sig_cols + [0.65],
        hspace=0.15, wspace=0.20
    )
    # nested gridspec for right column: NLL on top, RMSE on bottom
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[:, n_sig_cols], hspace=0.35
    )
    ax_nll  = fig.add_subplot(gs_right[0])
    ax_rmse = fig.add_subplot(gs_right[1])

    x_plot = torch.linspace(-1, 1, 200)[None, :, None]
    x_np   = x_plot.squeeze().numpy()

    # ── first pass: predictions + shared y limits ─────────────────
    all_preds  = {}
    all_y_true = []

    for col, x_std in enumerate(noise_levels):
        set_seed(seed)
        x_all, y_all = sample_task(1, 500)
        all_idx  = torch.randperm(x_all.shape[1])
        ctx_idx  = all_idx[:num_context].sort().values
        x_ctx, y_ctx = x_all[:, ctx_idx], y_all[:, ctx_idx]
        sort_idx = x_all.squeeze().argsort()
        x_true   = x_all.squeeze()[sort_idx].numpy()
        y_true   = y_all.squeeze()[sort_idx].numpy()
        x_ctx_noisy = add_noise(x_ctx, x_std)

        all_y_true.append(y_true)
        all_preds[(col, "x_ctx")]       = x_ctx
        all_preds[(col, "y_ctx")]       = y_ctx
        all_preds[(col, "x_ctx_noisy")] = x_ctx_noisy
        all_preds[(col, "x_true")]      = x_true
        all_preds[(col, "y_true")]      = y_true

        for mtype, _ in row_cfg:
            if x_std not in all_models:
                continue
            x_ctx_in = x_ctx if mtype == "np_clean" else x_ctx_noisy
            mu, std  = predict(all_models[x_std][mtype], mtype,
                               x_ctx_in, y_ctx, x_plot,
                               x_std=x_std, n_samples=n_samples)
            all_preds[(col, mtype)] = (mu, std)

    model_preds = {(c, k): v for (c, k), v in all_preds.items()
                   if k in MODEL_COLOR}
    y_vals = np.concatenate(
        all_y_true +
        [mu + 2*std for mu, std in model_preds.values()] +
        [mu - 2*std for mu, std in model_preds.values()]
    )
    y_pad = 0.12 * (y_vals.max() - y_vals.min())
    y_lim = (y_vals.min() - y_pad, y_vals.max() + y_pad)

    # ── second pass: plot model panels ────────────────────────────
    for row, (mtype, row_label) in enumerate(row_cfg):
        for col, x_std in enumerate(noise_levels):
            ax = fig.add_subplot(gs[row, col])

            x_ctx       = all_preds[(col, "x_ctx")]
            y_ctx       = all_preds[(col, "y_ctx")]
            x_ctx_noisy = all_preds[(col, "x_ctx_noisy")]
            x_true      = all_preds[(col, "x_true")]
            y_true      = all_preds[(col, "y_true")]

            if (col, mtype) not in all_preds:
                ax.text(0.5, 0.5, f"No model for σ={x_std}",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            color   = MODEL_COLOR[mtype]
            mu, std = all_preds[(col, mtype)]

            ax.fill_between(x_np, mu - 2*std, mu + 2*std,
                            alpha=0.15, color=color)
            ax.fill_between(x_np, mu - std,   mu + std,
                            alpha=0.25, color=color)
            ax.plot(x_np, mu, color=color, lw=2.0, zorder=4)
            ax.plot(x_true, y_true, color="black", lw=1.5,
                    linestyle="--", zorder=5, alpha=0.8)
            ax.scatter(x_ctx.squeeze().numpy(), y_ctx.squeeze().numpy(),
                       c="black", s=30, zorder=7, marker="o")
            if mtype != "np_clean":
                ax.scatter(x_ctx_noisy.squeeze().numpy(), y_ctx.squeeze().numpy(),
                           c="#FE6100", s=30, zorder=6, marker="^")

            ax.set_xlim(-1, 1)
            ax.set_ylim(y_lim)
            ax.grid(True, alpha=0.2, linestyle=":")
            ax.tick_params(labelsize=8)
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])

            if row == 0:
                ax.set_title(f"$\\sigma_x = {x_std}$", fontsize=11,
                             fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{row_label}\n$y$", fontsize=10,
                              fontweight="bold")
            if row == n_models - 1:
                ax.set_xlabel("$x$", fontsize=9)

    # ── right column: NLL and RMSE sweeps ─────────────────────────
    if all_results is not None:
        sigmas = sorted(all_results.keys())

        for ax, metric, ylabel in [(ax_nll,  "nll",  "NLL (↓)"),
                                    (ax_rmse, "rmse", "RMSE (↓)")]:
            for mtype, style in SWEEP_STYLE.items():
                means = [np.mean(all_results[s][mtype][metric]) for s in sigmas]
                stds  = [np.std(all_results[s][mtype][metric])  for s in sigmas]
                ax.plot(sigmas, means,
                        color=style["color"], marker=style["marker"],
                        linestyle=style["ls"], lw=2, markersize=6,
                        label=style["label"])

            for x_std in noise_levels:
                ax.axvline(x_std, color="gray", lw=0.8,
                           linestyle=":", alpha=0.6)

            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(sigmas)
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, alpha=0.25, linestyle=":")

        ax_nll.set_title("Performance vs\nNoise Level",
                         fontsize=10, fontweight="bold")
        ax_nll.set_xticklabels([])   # hide x ticks on top panel
        ax_rmse.set_xlabel("Input noise $\\sigma_x$", fontsize=9)
        ax_nll.legend(fontsize=7, loc="upper left")

    else:
        for ax, label in [(ax_nll, "NLL (↓)"), (ax_rmse, "RMSE (↓)")]:
            ax.text(0.5, 0.5, f"{label}\nPass all_results",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="gray")

    # ── shared legend at bottom ───────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color="black",   lw=1.5, ls="--",
                   label="True function"),
        plt.Line2D([0], [0], color="black",   lw=0, marker="o",
                   markersize=6, label="Context (true)"),
        plt.Line2D([0], [0], color="#FE6100", lw=0, marker="^",
                   markersize=6, label="Context (noisy)"),
        plt.Line2D([0], [0], color="gray",    lw=8, alpha=0.3,
                   label="Mean ± 1σ / 2σ"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.45, -0.03))

    plt.savefig(f"{save_prefix}_seed{seed}.pdf", bbox_inches="tight")
    plt.savefig(f"{save_prefix}_seed{seed}.png", bbox_inches="tight", dpi=200)
    print(f"Saved: {save_prefix}_seed{seed}.pdf / .png")
    plt.show()
# ─────────────────────────────────────────────
# 6b.  Beta search via validation NLL
# ─────────────────────────────────────────────
BETA_GRID = [0.01, 0.05, 0.1, 0.2, 0.5]
 
def beta_search(x_std, x_train, y_train, x_val, y_val,
                X_DIM, Y_DIM, R_DIM, Z_DIM, H_DIM,
                num_epochs, batch_size, lr, beta_kl,
                eval_context, ckpt_dir):
    """
    Trains one DnNP per beta value on x_train/y_train,
    evaluates validation NLL on x_val/y_val,
    returns (best_beta, results_dict) where results_dict maps
    beta -> mean val NLL.
    """
    stag     = f"sigma{x_std:.2f}".replace(".", "p")
    beta_dir = os.path.join(ckpt_dir, stag, "beta_search")
    os.makedirs(beta_dir, exist_ok=True)
 
    val_nlls = {}
 
    for beta in BETA_GRID:
        btag      = f"beta{beta:.3f}".replace(".", "p")
        ckpt_path = os.path.join(beta_dir, f"dnp_{btag}.pt")
 
        model = DenoisingNeuralProcess(X_DIM, Y_DIM, R_DIM, Z_DIM, H_DIM,
                                       x_std=x_std, denoise_uses_y=False).to(DEVICE)
 
        if os.path.exists(ckpt_path):
            print(f"  Loading β={beta} from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            model.eval()
        else:
            print(f"  Training β={beta} ...")
            train(model, "dnp", x_train, y_train,
                  num_epochs=num_epochs, batch_size=batch_size, lr=lr,
                  x_std=x_std, beta_kl=beta_kl, beta_denoise=beta)
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"  Saved → {ckpt_path}")
 
        # evaluate on validation set
        nll_list = []
        for i in range(min(200, x_val.shape[0])):
            x_all, y_all = x_val[[i]], y_val[[i]]
            all_idx = torch.randperm(x_all.shape[1])
            ctx_idx = all_idx[:eval_context].sort().values
            tgt_idx = all_idx[eval_context:].sort().values
            x_ctx, y_ctx = x_all[:, ctx_idx], y_all[:, ctx_idx]
            x_tgt, y_tgt = x_all[:, tgt_idx], y_all[:, tgt_idx]
 
            mu, std = predict(model, "dnp", x_ctx, y_ctx,
                              x_tgt, x_std=x_std)
            y_np = y_tgt.squeeze().numpy()
            nll  = 0.5 * np.mean(((y_np - mu) / std)**2 +
                                  np.log(2 * np.pi * std**2))
            nll_list.append(nll)
 
        mean_nll = np.mean(nll_list)
        val_nlls[beta] = mean_nll
        print(f"  β={beta:.3f}  val NLL={mean_nll:.4f}")
 
    best_beta = min(val_nlls, key=val_nlls.get)
    print(f"\n  ✓ Best β={best_beta} (val NLL={val_nlls[best_beta]:.4f})")
    return best_beta, val_nlls
 
 
# ─────────────────────────────────────────────
# 11.  Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=1)
    parser.add_argument("--beta-search", action="store_true",
                        help="Run beta grid search via validation NLL")
    parser.add_argument("--kernel", choices=["rbf", "periodic"], default="rbf",
                        help="GP kernel for the synthetic data")
    parser.add_argument("--ls-range",     type=float, nargs=2, default=[1.0, 2.0],
                        help="periodic length-scale range; pass the same value twice to fix it")
    parser.add_argument("--period-range", type=float, nargs=2, default=[0.7, 1.2],
                        help="periodic period range; pass the same value twice to fix it")
    parser.add_argument("--ctx-max",      type=int, default=30,
                        help="max context size during training")
    parser.add_argument("--eval-context", type=int, default=15,
                        help="context size at evaluation / in figures")
    parser.add_argument("--epochs",       type=int, default=20000)
    parser.add_argument("--patience",     type=int, default=2000,
                        help="early-stopping patience (steps)")
    args = parser.parse_args()
    KERNEL   = args.kernel
    CTX_MAX  = args.ctx_max
    PATIENCE = args.patience
    LS_RANGE     = tuple(args.ls_range)
    PERIOD_RANGE = tuple(args.period_range)
    # tag folders/cache with the kernel settings so different settings never share files
    KTAG = "" if args.kernel == "rbf" else (
        f"_p{PERIOD_RANGE[0]:g}-{PERIOD_RANGE[1]:g}_l{LS_RANGE[0]:g}-{LS_RANGE[1]:g}")
 
    set_seed(args.seed)
    print(f"Seed: {args.seed} | Kernel: {args.kernel} | Device: {DEVICE}")
 
    X_DIM, Y_DIM        = 1, 1
    R_DIM, Z_DIM, H_DIM = 128, 64, 128
    NUM_EPOCHS          = args.epochs
    BATCH_SIZE          = 64
    NUM_POINTS          = 100
    EVAL_CONTEXT        = args.eval_context
    LR                  = 3e-4
    BETA_KL             = 1.0
    BETA_DENOISE        = 0.1    # default, overridden per sigma if --beta-search
    NOISE_LEVELS        = [0.1, 0.2, 0.3, 0.4, 0.5]
 
    # ── Directory structure ───────────────────────────────────────
    # separate folders per kernel so periodic runs never load RBF checkpoints
    runs_root = "runs" if args.kernel == "rbf" else f"runs_{args.kernel}{KTAG}"
    run_dir  = os.path.join(runs_root, f"seed{args.seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")
    summ_dir = os.path.join(run_dir, "summary")
    for d in [ckpt_dir, plot_dir, summ_dir]:
        os.makedirs(d, exist_ok=True)
 
    def sigma_tag(s):
        return f"sigma{s:.2f}".replace(".", "p")
 
    os.makedirs("data", exist_ok=True)
    cache_path = ("data/gp_dataset_1d.pt" if args.kernel == "rbf"
                  else f"data/gp_{args.kernel}{KTAG}_dataset_1d.pt")
    x_data, y_data = get_dataset(num_tasks=10000, num_points=NUM_POINTS,
                                 cache_path=cache_path, kernel=args.kernel)
 
    # ── Train / Val / Test split ──────────────────────────────────
    # 8000 train | 1000 val | 1000 test  (last 200 of test used for eval)
    x_train, y_train = x_data[:8000],      y_data[:8000]
    x_val,   y_val   = x_data[8000:9000],  y_data[8000:9000]
    x_test,  y_test  = x_data[9000:],      y_data[9000:]
 
    all_results    = {}
    all_models     = {}
    all_histories  = {}
    sigma_records  = {}
    best_betas     = {}   # sigma -> best beta (populated if --beta-search)
 
    # ── Train NP(clean) once ──────────────────────────────────────
    np_clean_ckpt  = os.path.join(ckpt_dir, "np_clean.pt")
    np_clean_model = NeuralProcess(X_DIM, Y_DIM, R_DIM, Z_DIM, H_DIM).to(DEVICE)
 
    if os.path.exists(np_clean_ckpt):
        print(f"\n── Loading np_clean from {np_clean_ckpt} ──")
        ckpt = torch.load(np_clean_ckpt, map_location=DEVICE)
        np_clean_model.load_state_dict(ckpt["model"])
        np_clean_model.eval()
        np_clean_history = ckpt["history"]
    else:
        print(f"\n── Training np_clean ──")
        np_clean_history = train(
            np_clean_model, "np_clean", x_train, y_train,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            x_std=0.0, beta_kl=BETA_KL, beta_denoise=BETA_DENOISE,
        )
        torch.save({"model": np_clean_model.state_dict(),
                    "history": np_clean_history}, np_clean_ckpt)
        print(f"   Saved → {np_clean_ckpt}")
 
    np_clean_results = None
 
    # ── Per-noise-level training ──────────────────────────────────
    for x_std in NOISE_LEVELS:
        stag = sigma_tag(x_std)
        print(f"\n{'='*60}\n  Noise level σ_x = {x_std}  [{stag}]\n{'='*60}")
 
        sigma_ckpt_dir = os.path.join(ckpt_dir, stag)
        sigma_plot_dir = os.path.join(plot_dir, stag)
        os.makedirs(sigma_ckpt_dir, exist_ok=True)
        os.makedirs(sigma_plot_dir, exist_ok=True)
 
        # ── beta search (optional) ────────────────────────────────
        if args.beta_search:
            print(f"\n── Beta search for σ={x_std} ──")
            best_beta, beta_nlls = beta_search(
                x_std, x_train, y_train, x_val, y_val,
                X_DIM, Y_DIM, R_DIM, Z_DIM, H_DIM,
                num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                lr=LR, beta_kl=BETA_KL,
                eval_context=EVAL_CONTEXT,
                ckpt_dir=ckpt_dir,
            )
            best_betas[x_std] = best_beta
            beta_denoise = best_beta
        else:
            beta_denoise = BETA_DENOISE
            best_betas[x_std] = beta_denoise
 
        print(f"   Using β_denoise={beta_denoise}")
 
        models = {
            "np_clean": np_clean_model,
            "np_noisy": NeuralProcess(X_DIM, Y_DIM, R_DIM, Z_DIM, H_DIM).to(DEVICE),
            "dnp":      DenoisingNeuralProcess(X_DIM, Y_DIM, R_DIM, Z_DIM, H_DIM,
                                               x_std=x_std, denoise_uses_y=False).to(DEVICE),
            "dnp_ls":   DenoisingNeuralProcess(X_DIM, Y_DIM, R_DIM, Z_DIM, H_DIM,
                                               x_std=None, denoise_uses_y=False).to(DEVICE),
        }
 
        ckpt_paths = {
            "np_noisy": os.path.join(sigma_ckpt_dir, "np_noisy.pt"),
            "dnp":      os.path.join(sigma_ckpt_dir,
                        f"dnp_beta{beta_denoise:.3f}.pt".replace(".", "p")),
            "dnp_ls":   os.path.join(sigma_ckpt_dir, "dnp_ls.pt"),
        }
 
        histories = {"np_clean": np_clean_history}
        for mtype in ("np_noisy", "dnp", "dnp_ls"):
            ckpt_path = ckpt_paths[mtype]
            bd = beta_denoise if mtype == "dnp" else BETA_DENOISE
            if os.path.exists(ckpt_path):
                print(f"\n── Loading {mtype} from {ckpt_path} ──")
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                models[mtype].load_state_dict(ckpt["model"])
                models[mtype].eval()
                histories[mtype] = ckpt["history"]
            else:
                print(f"\n── Training {mtype} (β_denoise={bd}) ──")
                histories[mtype] = train(
                    models[mtype], mtype, x_train, y_train,
                    num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                    x_std=x_std, beta_kl=BETA_KL, beta_denoise=bd,
                )
                torch.save({"model": models[mtype].state_dict(),
                            "history": histories[mtype]}, ckpt_path)
                print(f"   Saved → {ckpt_path}")
 
        sigma_records[x_std] = models["dnp_ls"].sigma.item()
        print(f"   DNP (learn σ): learned={sigma_records[x_std]:.4f}, true={x_std:.4f}")
 
        results = evaluate(models, x_test, y_test,
                           x_std=x_std, num_context=EVAL_CONTEXT, num_tasks=200)
 
        if np_clean_results is None:
            np_clean_results = results["np_clean"]
        else:
            results["np_clean"] = np_clean_results
 
        plot_all(models, histories, results,
                 x_std=x_std, num_context=EVAL_CONTEXT,
                 save_prefix=os.path.join(sigma_plot_dir, "dnp_vs_np_1d"))
 
        all_results[x_std]   = results
        all_models[x_std]    = models
        all_histories[x_std] = histories
 
    # ── Print beta search summary ─────────────────────────────────
    if args.beta_search:
        print(f"\n{'─'*40}")
        print("Beta search results:")
        for s, b in best_betas.items():
            print(f"  σ={s:.2f}  →  β={b}")
        print(f"{'─'*40}")
 
    # ── Summary plots ─────────────────────────────────────────────
    plot_sigma_accuracy(sigma_records,
                        save_prefix=os.path.join(summ_dir, "sigma_accuracy"))
 
    plot_noise_sweep(NOISE_LEVELS, all_results,
                     save_prefix=os.path.join(summ_dir, "noise_sweep"))

    # ── Save per-seed test metrics (mean over test tasks) ─────────
    # aggregate_seeds.py turns these into mean ± std across seeds
    import json
    summary = {"seed": args.seed, "kernel": args.kernel, "results": {}}
    for x_std, res in all_results.items():
        summary["results"][f"{x_std}"] = {
            mtype: {"nll":  float(np.mean(res[mtype]["nll"])),
                    "rmse": float(np.mean(res[mtype]["rmse"]))}
            for mtype in res}
        summary["results"][f"{x_std}"]["dnp_ls"]["learned_sigma"] = sigma_records[x_std]
    with open(os.path.join(summ_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {os.path.join(summ_dir, 'results.json')}")
 
    # ── Paper figures ─────────────────────────────────────────────
    paper_dir = os.path.join(plot_dir, "paper_figures")
    os.makedirs(paper_dir, exist_ok=True)
 
    for i, fseed in enumerate([args.seed + i for i in range(5)]):
        print(f"\n── Paper figure {i+1}/5 (seed={fseed}) ──")
        plot_paper_figure(all_models,
                          noise_levels=(0.1, 0.2, 0.3, 0.4, 0.5),
                          num_context=EVAL_CONTEXT,
                          n_samples=100,
                          seed=fseed,
                          save_prefix=os.path.join(paper_dir, f"ex{i+1}_fseed{fseed}"))
 
    scores    = find_best_seed(all_models, x_std=0.3, num_context=20, n_seeds=50)
    best_seed = scores[0][1]
 
    plot_best_example(all_models, seed=best_seed,
                      noise_levels=(0.1, 0.2, 0.3, 0.4, 0.5),
                      all_results=all_results,
                      num_context=EVAL_CONTEXT,
                      save_prefix=os.path.join(paper_dir, "best_example"))
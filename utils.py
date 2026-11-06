import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from pathlib import Path
import sys
import json
from neural_process import DenoisingNeuralProcessImg
import logging
from torch.distributions.kl import kl_divergence

import random

def create_random_masks(img_shape, num_context, num_target, device):
    """Create random pixel masks."""
    batch_size, channels, height, width = img_shape
    
    context_mask = torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)
    target_mask = torch.zeros(batch_size, height, width, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        idx = torch.randperm(height * width)
        context_mask[b].view(-1)[idx[:num_context]] = True
        target_mask[b].view(-1)[idx[:num_target]] = True
    
    return context_mask, target_mask


def img_mask_to_np_input(img, mask, normalize=True):
    """Convert image + mask to NP input format."""
    batch_size, num_channels, height, width = img.size()
    mask_img_size = mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    num_points = mask[0].nonzero().size(0)
    nonzero_idx = mask.nonzero()
    
    x = nonzero_idx[:, 1:].view(batch_size, num_points, 2).float()
    y = img[mask_img_size].view(batch_size, num_channels, num_points)
    y = y.permute(0, 2, 1)

    if normalize:
        x = (x - float(height) / 2) / (float(height) / 2)
        y -= 0.5

    return x, y

def chosen_context_target_split(x, y, context_list, target_list):
    """Given inputs x and their value y, return chosen subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points

    c2 = random.sample(context_list, len(context_list))
    t2 = random.sample(target_list, len(target_list))

    x_context = x[:, c2, :]
    y_context = y[:, c2, :]
    x_target = x[:, t2, :]
    y_target = y[:, t2, :]
    return x_context, y_context, x_target, y_target


def get_beta(epoch, warmup_epochs=20, anneal_epochs=50, beta_max=0.01):
    """Beta annealing schedule."""
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + anneal_epochs:
        progress = (epoch - warmup_epochs) / anneal_epochs
        return beta_max * progress  # Linear
    else:
        return beta_max


def evaluate_model_on_test(model, test_loader, noise_std, device, is_dnp=False):
    """Evaluate model on test set."""
    
    model.eval()
    total_nll = 0.
    total_rmse = 0.
    
    with torch.no_grad():
        for img, _ in test_loader:
            img = img.to(device)
            
            num_ctx = 100
            ctx_mask, _ = create_random_masks(img.shape, num_ctx, 0, device)
            tgt_mask = torch.ones_like(ctx_mask)
            
            x_ctx_clean, y_ctx = img_mask_to_np_input(img, ctx_mask, normalize=True)
            x_tgt_clean, y_tgt = img_mask_to_np_input(img, tgt_mask, normalize=True)
            
            # Add noise if needed
            if noise_std > 0:
                x_ctx_noisy = x_ctx_clean + torch.randn_like(x_ctx_clean) * noise_std
                x_ctx_noisy = torch.clamp(x_ctx_noisy, -1, 1)
                x_ctx_input = x_ctx_noisy
            else:
                x_ctx_input = x_ctx_clean
            
            # Forward (force training mode for tuple output)
            model.train()
            if is_dnp:
                p_y, _, _, _ = model(x_ctx_input, y_ctx, x_tgt_clean, y_tgt)
            else:
                p_y, _, _ = model.neural_process(x_ctx_input, y_ctx, x_tgt_clean, y_tgt)
            model.eval()
            
            # Metrics
            log_likelihood = p_y.log_prob(y_tgt).mean(dim=0).sum()
            nll = -log_likelihood / y_tgt.numel()
            
            y_pred = p_y.mean + 0.5
            rmse = torch.sqrt(((y_pred - (y_tgt + 0.5)) ** 2).mean())
            
            total_nll += nll.item()
            total_rmse += rmse.item()
    
    return {
        'nll': total_nll / len(test_loader),
        'rmse': total_rmse / len(test_loader)
    }



class ExperimentTracker:
    def __init__(self, exp_dir):
        self.exp_dir = Path(exp_dir)
        self.metrics = {
            'train': {'loss': [], 'logp': [], 'kl_z': [], 'kl_x': [],
                     'beta': [], 'correction_ratio': []},
            'val': {'loss': [], 'logp': [], 'kl_z': [], 'rmse': [], 'nll': []},
            'test': {},
            'best': {'epoch': 0, 'val_nll': float('inf')}  # ⭐ Changed
        }
    
    def log_epoch(self, epoch, phase, metrics_dict):
        for key, value in metrics_dict.items():
            if key in self.metrics[phase]:
                self.metrics[phase][key].append(value)
        
        if phase == 'val' and 'nll' in metrics_dict:
            if metrics_dict['nll'] < self.metrics['best']['val_nll']:  # ⭐ Changed
                self.metrics['best']['val_nll'] = metrics_dict['nll']
                self.metrics['best']['epoch'] = epoch
                self.metrics['best']['metrics'] = metrics_dict
                return True
        return False

    def log_test(self, metrics_dict):
        self.metrics['test'] = metrics_dict
    
    def save(self):
        with open(self.exp_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)


def train_epoch(model, loader, epoch, config, device, logger):
    """Train one epoch."""
    
    model.train()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get('lr', 5e-5),
        weight_decay=config.get('weight_decay', 0)
    )
    
    beta_x = get_beta(
        epoch,
        warmup_epochs=config.get('beta_warmup', 40),
        anneal_epochs=config.get('beta_anneal', 50),
        beta_max=config.get('beta_final', 0.01)
    )
    
    epoch_loss = 0.
    epoch_logp = 0.
    epoch_kl_z = 0.
    epoch_kl_x = 0.
    total_correction = 0.
    total_noise = 0.
    n_batches = 0
    
    noise_std = config['noise_std']
    
    for img, _ in loader:
        img = img.to(device)
        
        num_ctx = np.random.randint(3, 200)
        ctx_mask, _ = create_random_masks(img.shape, num_ctx, 0, device)
        tgt_mask = torch.ones_like(ctx_mask)
        
        x_ctx_clean, y_ctx = img_mask_to_np_input(img, ctx_mask, normalize=True)
        x_tgt_clean, y_tgt = img_mask_to_np_input(img, tgt_mask, normalize=True)
        
        x_ctx_noisy = x_ctx_clean + torch.randn_like(x_ctx_clean) * noise_std
        x_ctx_noisy = torch.clamp(x_ctx_noisy, -1, 1)
        
        optimizer.zero_grad()
        p_y, q_z_tgt, q_z_ctx, q_xs_ctx = model(
            x_ctx_noisy, y_ctx, x_tgt_clean, y_tgt
        )
        
        with torch.no_grad():
            x_denoised_mean = q_xs_ctx.mean
            noise_correction = (x_denoised_mean - x_ctx_noisy).abs().mean().item()
            true_noise = (x_ctx_clean - x_ctx_noisy).abs().mean().item()
            if not hasattr(train_epoch, 'denoised_range'):
                train_epoch.denoised_range = (x_denoised_mean.min().item(), x_denoised_mean.max().item())

        log_likelihood = p_y.log_prob(y_tgt).mean(dim=0).sum()
        kl_z = kl_divergence(q_z_tgt, q_z_ctx).mean(dim=0).sum()
        
        p_x_ctx = Normal(x_ctx_noisy, noise_std)
        kl_x = kl_divergence(q_xs_ctx, p_x_ctx).mean(dim=0).sum()
        
        loss = -log_likelihood + kl_z + beta_x * kl_x
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_logp += log_likelihood.item()
        epoch_kl_z += kl_z.item()
        epoch_kl_x += kl_x.item()
        total_correction += noise_correction
        total_noise += true_noise
        n_batches += 1
    
    return {
        'loss': epoch_loss / n_batches,
        'logp': epoch_logp / n_batches,
        'kl_z': epoch_kl_z / n_batches,
        'kl_x': epoch_kl_x / n_batches,
        'beta': beta_x,
        'correction_ratio': total_correction / total_noise if total_noise > 0 else 0.0,
        'denoised_range': getattr(train_epoch, 'denoised_range', (0, 0)) 
    }


def validate_epoch(model, loader, config, device):
    """Validate one epoch."""
    
    model.eval()
    
    val_loss = 0.
    val_logp = 0.
    val_kl_z = 0.
    val_rmse = 0.
    val_nll = 0.
    n_batches = 0
    
    noise_std = config['noise_std']
    beta = config.get('beta_final', 0.01)
    
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device)
            
            num_ctx = 100
            ctx_mask, _ = create_random_masks(img.shape, num_ctx, 0, device)
            tgt_mask = torch.ones_like(ctx_mask)
            
            x_ctx_clean, y_ctx = img_mask_to_np_input(img, ctx_mask, normalize=True)
            x_tgt_clean, y_tgt = img_mask_to_np_input(img, tgt_mask, normalize=True)
            
            x_ctx_noisy = x_ctx_clean + torch.randn_like(x_ctx_clean) * noise_std
            x_ctx_noisy = torch.clamp(x_ctx_noisy, -1, 1)
            
            model.train()
            p_y, q_z_tgt, q_z_ctx, q_xs_ctx = model(
                x_ctx_noisy, y_ctx, x_tgt_clean, y_tgt
            )
            model.eval()
            
            log_likelihood = p_y.log_prob(y_tgt).mean(dim=0).sum()
            kl_z = kl_divergence(q_z_tgt, q_z_ctx).mean(dim=0).sum()
            
            p_x_ctx = Normal(x_ctx_noisy, noise_std)
            kl_x = kl_divergence(q_xs_ctx, p_x_ctx).mean(dim=0).sum()
            
            loss = -log_likelihood + kl_z + beta * kl_x
            
            y_pred_mean = p_y.mean + 0.5
            rmse = torch.sqrt(((y_pred_mean - (y_tgt + 0.5)) ** 2).mean())
            nll = -log_likelihood / y_tgt.numel()
            
            val_loss += loss.item()
            val_logp += log_likelihood.item()
            val_kl_z += kl_z.item()
            val_rmse += rmse.item()
            val_nll += nll.item()
            n_batches += 1
    
    return {
        'loss': val_loss / n_batches,
        'logp': val_logp / n_batches,
        'kl_z': val_kl_z / n_batches,
        'rmse': val_rmse / n_batches,
        'nll': val_nll / n_batches
        }


def run_single_experiment(config, exp_id, base_dir='results'):
    """Run one DNP experiment with full logging and checkpointing."""
    
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import traceback
    
    # Setup directories
    exp_dir = Path(base_dir) / f'exp_{exp_id:04d}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # ⭐ Create a UNIQUE logger for this experiment
    logger = logging.getLogger(f'exp_{exp_id}')  # Unique name!
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # File handler
    fh = logging.FileHandler(log_dir / f'exp_{exp_id:04d}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Starting experiment {exp_id}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize tracker
    tracker = ExperimentTracker(exp_dir)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load data with CONSISTENT split
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data_full = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Same split everywhere
        val_size = len(test_data_full) // 2
        test_size = len(test_data_full) - val_size
        torch.manual_seed(42)
        val_data, test_data = torch.utils.data.random_split(test_data_full, [val_size, test_size])
        
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
        
        logger.info(f"Train: {len(train_data)} samples")
        logger.info(f"Val: {len(val_data)} samples")
        logger.info(f"Test: {len(test_data)} samples")
        
        # Create model
        logger.info("Creating model...")
        model = DenoisingNeuralProcessImg(
            img_size=tuple(config['img_size']),
            r_dim=config['r_dim'],
            z_dim=config['z_dim'],
            h_dim=config['h_dim'],
            x_std=config['noise_std'],
            dropout=config.get('dropout', 0.0),
            denoise_depth=config.get('denoise_depth', 3),
            denoise_width=config.get('denoise_width', 256),
            denoise_uses_y=config.get('denoise_uses_y', False)
        ).to(device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"denoise_uses_y={config.get('denoise_uses_y', False)}")
        
        # Train
        logger.info("Starting training...")
        
        for epoch in range(config['epochs']):
            # Train
            train_metrics = train_epoch(model, train_loader, epoch, config, device, logger)
            tracker.log_epoch(epoch, 'train', train_metrics)
            
            # Log
            if epoch % 5 == 0:
                d_min, d_max = train_metrics.get('denoised_range', (0, 0))
                logger.info(
                    f"Epoch {epoch}: "
                    f"Loss={train_metrics['loss']:.2f} | "
                    f"LogP={train_metrics['logp']:.2f} | "
                    f"KL_z={train_metrics['kl_z']:.2f} | "
                    f"KL_x={train_metrics['kl_x']:.2f} | "
                    f"β={train_metrics['beta']:.3f} | "
                    f"Correction={train_metrics['correction_ratio']:.2f} | "
                    f"Range=[{d_min:.2f},{d_max:.2f}]"

                )
            
            # Validate
            if epoch % 10 == 0 or epoch == config['epochs'] - 1:
                val_metrics = validate_epoch(model, val_loader, config, device)
                is_best = tracker.log_epoch(epoch, 'val', val_metrics)
                
                logger.info(f"  Val: NLL={val_metrics['nll']:.4f} | RMSE={val_metrics['rmse']:.4f}")
                
                if is_best:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'metrics': val_metrics
                    }, checkpoint_dir / 'best_model.pth')
                    logger.info(f"✓ New best! Val NLL: {val_metrics['nll']:.4f}")
            
            # Save periodically
            if epoch % 20 == 0:
                tracker.save()
        
        # Final test
        logger.info("Running final test evaluation...")
        test_metrics = validate_epoch(model, test_loader, config, device)
        tracker.log_test(test_metrics)
        
        logger.info(f"Test Results:")
        logger.info(f"  RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"  NLL: {test_metrics['nll']:.4f}")
        
        tracker.save()
        (exp_dir / 'COMPLETED').touch()
        logger.info("Experiment completed successfully!")
        
        return {
            'exp_id': exp_id,
            'status': 'success',
            'test_metrics': test_metrics,
            'best_epoch': tracker.metrics['best']['epoch']
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        (exp_dir / 'FAILED').touch()
        with open(exp_dir / 'error.txt', 'w') as f:
            f.write(traceback.format_exc())
        
        return {'exp_id': exp_id, 'status': 'failed', 'error': str(e)}
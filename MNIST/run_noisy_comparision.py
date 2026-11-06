"""
For one noise level:
1. Train Noisy NP baseline
2. Run DNP sweep (8 experiments)
3. Select best DNP by validation NLL
4. Compare: Clean NP vs Noisy NP vs Best DNP on test set
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from neural_process import NeuralProcessImg, DenoisingNeuralProcessImg
from torch.distributions.kl import kl_divergence
import numpy as np
from pathlib import Path
import json
import pandas as pd
import sys

# Import helpers
from utils import create_random_masks, img_mask_to_np_input, run_single_experiment, evaluate_model_on_test, load_mnist_classifier



def train_noisy_np(noise_std, save_dir):
    """Train Noisy NP baseline."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_file = Path(save_dir) / 'noisy_np.pth'
    
    # Check if exists
    if save_file.exists():
        print(f"✓ Noisy NP already trained: {save_file}")
        checkpoint = torch.load(save_file, weights_only=False)
        return checkpoint
    
    print(f"\nTraining Noisy NP baseline (σ={noise_std})...")
    print("="*60)
    
    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    
    # Create model
    model = NeuralProcessImg(
        img_size=(1, 28, 28), 
        r_dim=256, 
        z_dim=256, 
        h_dim=256
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    
    history = {'loss': [], 'logp': [], 'kl': []}
    
    # Early stopping
    best_val_nll = float('inf')
    patience = 15
    no_improve = 0
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0.
        epoch_logp = 0.
        epoch_kl = 0.
        
        for img, _ in train_loader:
            img = img.to(device, non_blocking=True)
            
            # Fixed context size for consistency (can still vary slightly)
            num_ctx = np.random.randint(50, 150)
            # KEY OPTIMIZATION: Use subset of target pixels instead of all 784
            num_tgt = 300  # Much faster than 784, still good coverage
            ctx_mask, tgt_mask = create_random_masks(img.shape, num_ctx, num_tgt, device)
            
            x_ctx, y_ctx = img_mask_to_np_input(img, ctx_mask, normalize=True)
            x_tgt, y_tgt = img_mask_to_np_input(img, tgt_mask, normalize=True)
            
            # Add noise to context
            x_ctx_noisy = x_ctx + torch.randn_like(x_ctx) * noise_std
            x_ctx_noisy = torch.clamp(x_ctx_noisy, -1, 1)
            
            optimizer.zero_grad()
            p_y, q_z_tgt, q_z_ctx = model.neural_process(
                x_ctx_noisy, y_ctx, x_tgt, y_tgt
            )
            
            log_likelihood = p_y.log_prob(y_tgt).mean(dim=0).sum()
            kl = kl_divergence(q_z_tgt, q_z_ctx).mean(dim=0).sum()
            loss = -log_likelihood + kl
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_logp += log_likelihood.item()
            epoch_kl += kl.item()
        
        history['loss'].append(epoch_loss / len(train_loader))
        history['logp'].append(epoch_logp / len(train_loader))
        history['kl'].append(epoch_kl / len(train_loader))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: LogP={history['logp'][-1]:.2f} | KL={history['kl'][-1]:.2f}")
        
        # Simple validation for early stopping (every 5 epochs)
        if epoch % 5 == 0 and epoch > 20:
            model.eval()
            val_nll = 0.
            val_batches = 0
            with torch.no_grad():
                for img, _ in train_loader:
                    if val_batches >= 20:  # Quick validation on 20 batches
                        break
                    img = img.to(device, non_blocking=True)
                    num_ctx = 100
                    num_tgt = 200
                    ctx_mask, tgt_mask = create_random_masks(img.shape, num_ctx, num_tgt, device)
                    x_ctx, y_ctx = img_mask_to_np_input(img, ctx_mask, normalize=True)
                    x_tgt, y_tgt = img_mask_to_np_input(img, tgt_mask, normalize=True)
                    x_ctx_noisy = x_ctx + torch.randn_like(x_ctx) * noise_std
                    x_ctx_noisy = torch.clamp(x_ctx_noisy, -1, 1)
                    
                    # In eval mode, need to use train mode to get tuple output
                    model.train()
                    p_y, _, _ = model.neural_process(x_ctx_noisy, y_ctx, x_tgt, y_tgt)
                    model.eval()
                    
                    nll = -p_y.log_prob(y_tgt).sum() / y_tgt.numel()  # per-sample NLL
                    val_nll += nll.item()
                    val_batches += 1
            
            val_nll /= val_batches
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (val_nll={val_nll:.4f})")
                    break
    
    print(f" Final LogP: {history['logp'][-1]:.2f}")
    
    # Save
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {'epochs': 100, 'noise_std': noise_std}
    }
    
    torch.save(checkpoint, save_file)
    print(f" Saved to: {save_file}\n")
    
    return checkpoint


def run_dnp_sweep(noise_std, config_file, results_dir):
    """Run DNP hyperparameter sweep."""
    
    print(f"\nRunning DNP sweep (σ={noise_std})...")
    print("="*60)
    
    # Generate config if it doesn't exist
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"Config file not found: {config_file}")
        print("Generating config file...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent / 'configs'))
        from generate_configs import generate_configs_for_noise
        configs = generate_configs_for_noise(noise_std)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(configs, f, indent=2)
        print(f"Generated {config_file}: {len(configs)} experiments")
    else:
        # Load configs
        with open(config_file) as f:
            configs = json.load(f)
    
    n_experiments = len(configs)
    print(f"Experiments to run: {n_experiments}\n")
    
    # Run each experiment
    for i, config in enumerate(configs):
        exp_dir = Path(results_dir) / f'exp_{i:04d}'
        
        # Skip if completed
        if (exp_dir / 'COMPLETED').exists():
            print(f"Skipping exp {i} (already completed)")
            continue
        
        print(f"Running experiment {i+1}/{n_experiments}...")
        result = run_single_experiment(config, i, results_dir)
        print(f"  Status: {result['status']}\n")


def select_best_and_compare(noise_std, results_dir):
    """
    Select best DNP by validation NLL.
    Evaluate on test set.
    Compare to baselines.
    """
    
    results_dir = Path(results_dir)
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS FOR NOISE σ={noise_std}")
    print(f"{'='*70}\n")
    
    # Load DNP sweep results
    print("Loading DNP sweep results...")
    sweep_results = []
    
    for exp_dir in sorted(results_dir.glob('exp_*')):
        if not (exp_dir / 'COMPLETED').exists():
            continue
        
        try:
            with open(exp_dir / 'config.json') as f:
                config = json.load(f)
            with open(exp_dir / 'metrics.json') as f:
                metrics = json.load(f)
            
            sweep_results.append({
                'exp_id': config['exp_id'],
                'exp_name': exp_dir.name,
                # 'dropout': config['dropout'],
                'beta_warmup': config['beta_warmup'],
                'beta_final': config['beta_final'],
                'best_val_nll': metrics['best']['val_nll'],
                'test_nll': metrics['test']['nll'],
                'test_rmse': metrics['test']['rmse'],
            })
        except Exception as e:
            print(f"Warning: Failed to load {exp_dir}: {e}")
    
    if not sweep_results:
        print(" No completed DNP experiments found!")
        return
    
    df = pd.DataFrame(sweep_results)
    print(f"Loaded {len(df)} completed DNP experiments\n")
    
    #  Select best by VALIDATION NLL
    best_idx = df['best_val_nll'].idxmin()  # More negative = better
    best = df.loc[best_idx]
    
    print("All DNP experiments:")
    print(df[['exp_name',  'beta_warmup', 'beta_final', 
              'best_val_nll', 'test_nll']].to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"BEST DNP (selected by validation NLL):")
    print(f"{'='*70}")
    print(f"  Experiment: {best['exp_name']}")
    # print(f"  Dropout: {best['dropout']}")
    print(f"  Beta warmup: {best['beta_warmup']}")
    print(f"  Beta final: {best['beta_final']}")
    print(f"  Best Val NLL: {best['best_val_nll']:.4f} ⭐")
    print(f"  Test NLL: {best['test_nll']:.4f}")
    print(f"  Test RMSE: {best['test_rmse']:.4f}")
    
    # Load baselines for comparison
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load test data (same split as training!)
    transform = transforms.Compose([transforms.ToTensor()])
    test_data_full = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    val_size = len(test_data_full) // 2
    test_size = len(test_data_full) - val_size
    torch.manual_seed(42)  #  Same seed as training
    val_data, test_data = torch.utils.data.random_split(test_data_full, [val_size, test_size])
    
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # Load classifier for visual quality metric
    print("\nLoading MNIST classifier for visual quality metric...")
    classifier_model = load_mnist_classifier()
    
    # Evaluate Clean NP
    print(f"\n{'='*70}")
    print("EVALUATING BASELINES ON TEST SET:")
    print(f"{'='*70}")
    
    print("\nLoading Clean NP...")
    clean_checkpoint = torch.load('results/clean_np.pth', weights_only=False)
    clean_model = NeuralProcessImg(
        img_size=(1, 28, 28), r_dim=256, z_dim=256, h_dim=256
    ).to(device)
    clean_model.load_state_dict(clean_checkpoint['model_state_dict'])
    
    clean_test = evaluate_model_on_test(clean_model, test_loader, 0.0, device, is_dnp=False, classifier_model=classifier_model, eval_seed=42)
    print(f"  Clean NP: NLL={clean_test['nll']:.4f}, RMSE={clean_test['rmse']:.4f}, Classifier Acc={clean_test.get('classifier_acc', 0):.4f}")
    
    # Evaluate Noisy NP
    print(f"\nLoading Noisy NP (σ={noise_std})...")
    noisy_checkpoint = torch.load(results_dir / 'noisy_np.pth', weights_only=False)
    noisy_model = NeuralProcessImg(
        img_size=(1, 28, 28), r_dim=256, z_dim=256, h_dim=256
    ).to(device)
    noisy_model.load_state_dict(noisy_checkpoint['model_state_dict'])
    
    noisy_test = evaluate_model_on_test(noisy_model, test_loader, noise_std, device, is_dnp=False, classifier_model=classifier_model, eval_seed=42)
    print(f"  Noisy NP: NLL={noisy_test['nll']:.4f}, RMSE={noisy_test['rmse']:.4f}, Classifier Acc={noisy_test.get('classifier_acc', 0):.4f}")
    
    # Evaluate Best DNP with classifier metric
    print(f"\nEvaluating Best DNP (σ={noise_std})...")
    best_exp_dir = results_dir / best['exp_name']
    best_checkpoint = torch.load(best_exp_dir / 'checkpoints' / 'best_model.pth', weights_only=False)
    best_dnp_model = DenoisingNeuralProcessImg(
        img_size=(1, 28, 28),
        r_dim=best_checkpoint['config']['r_dim'],
        z_dim=best_checkpoint['config']['z_dim'],
        h_dim=best_checkpoint['config']['h_dim'],
        x_std=best_checkpoint['config']['noise_std'],
        denoise_uses_y=best_checkpoint['config'].get('denoise_uses_y', False)
    ).to(device)
    best_dnp_model.load_state_dict(best_checkpoint['model_state_dict'])
    
    best_dnp_test = evaluate_model_on_test(best_dnp_model, test_loader, noise_std, device, is_dnp=True, classifier_model=classifier_model, eval_seed=42)
    print(f"  Best DNP: NLL={best_dnp_test['nll']:.4f}, RMSE={best_dnp_test['rmse']:.4f}, Classifier Acc={best_dnp_test.get('classifier_acc', 0):.4f}")
    
    # Final comparison
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON (σ={noise_std}):")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Test NLL':<15} {'Test RMSE':<15} {'Classifier Acc':<15}")
    print("-"*70)
    print(f"{'Clean NP':<15} {clean_test['nll']:<15.4f} {clean_test['rmse']:<15.4f} {clean_test.get('classifier_acc', 0):<15.4f}")
    print(f"{'Noisy NP':<15} {noisy_test['nll']:<15.4f} {noisy_test['rmse']:<15.4f} {noisy_test.get('classifier_acc', 0):<15.4f}")
    print(f"{'Best DNP':<15} {best_dnp_test['nll']:<15.4f} {best_dnp_test['rmse']:<15.4f} {best_dnp_test.get('classifier_acc', 0):<15.4f}")
    print("-"*70)
    
    # Performance analysis
    dnp_vs_noisy_nll = best_dnp_test['nll'] - noisy_test['nll']
    dnp_vs_clean_nll = best_dnp_test['nll'] - clean_test['nll']
    dnp_vs_noisy_cls = best_dnp_test.get('classifier_acc', 0) - noisy_test.get('classifier_acc', 0)
    dnp_vs_clean_cls = best_dnp_test.get('classifier_acc', 0) - clean_test.get('classifier_acc', 0)
    
    print(f"\nPerformance vs baselines:")
    print(f"  DNP vs Noisy NP (NLL): {dnp_vs_noisy_nll:+.4f} {'DNP WINS' if dnp_vs_noisy_nll < 0 else ' Noisy wins'}")
    print(f"  DNP vs Clean NP (NLL): {dnp_vs_clean_nll:+.4f}")
    print(f"  DNP vs Noisy NP (Classifier Acc): {dnp_vs_noisy_cls:+.4f} {'DNP WINS' if dnp_vs_noisy_cls > 0 else ' Noisy wins'}")
    print(f"  DNP vs Clean NP (Classifier Acc): {dnp_vs_clean_cls:+.4f}")
    
    # Save summary
    summary = {
    'noise_std': float(noise_std),  #  Convert to Python float
    'clean_np': clean_test,
    'noisy_np': noisy_test,
    'best_dnp': {
        'exp_name': str(best['exp_name']),  #  Convert to string
        'config': {
            # 'dropout': float(best['dropout']),  #  Convert to float
            'beta_warmup': int(best['beta_warmup']),  #  Convert to int
            'beta_final': float(best['beta_final']),
        },
        'test': best_dnp_test
    },
    'all_dnp_results': df.to_dict('records')
}

    
    summary_file = results_dir / 'comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Saved summary to: {summary_file}")
    print("="*70 + "\n")
    
    return summary


def main(noise_std):
    """Complete pipeline for one noise level."""
    
    results_dir = Path(f'results/noise_{noise_std}')
    results_dir.mkdir(parents=True, exist_ok=True)
    config_file = f'configs/dnp_noise_{noise_std}.json'
    
    print("\n" + "="*70)
    print(f"COMPLETE PIPELINE FOR NOISE σ={noise_std}")
    print("="*70)
    
    # Step 1: Train Noisy NP
    print("\nStep 1: Train Noisy NP baseline")
    print("-"*70)
    train_noisy_np(noise_std, results_dir)
    
    # Step 2: Run DNP sweep
    print("\nStep 2: DNP Hyperparameter Sweep")
    print("-"*70)
    run_dnp_sweep(noise_std, config_file, results_dir)
    
    # Step 3: Compare
    print("\nStep 3: Select Best and Compare")
    print("-"*70)
    summary = select_best_and_compare(noise_std, results_dir)
    
    print("\n Pipeline complete for σ={noise_std}!")
    
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_and_compare_noise_level.py <noise_std>")
        print("Example: python run_and_compare_noise_level.py 0.15")
        sys.exit(1)
    
    noise_std = float(sys.argv[1])
    main(noise_std)
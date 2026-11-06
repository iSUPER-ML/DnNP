import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


class XStarEncoderMultiNoise(nn.Module):
    """
    Denoiser that handles different noise levels per dimension.
    
    Maps (x_noisy, r) → distribution over x_clean
    where different dimensions of x have different noise levels.
    """
    def __init__(self, x_dim, r_dim, h_dim, x_std_per_dim, uses_r=True):
        super().__init__()
        self.x_dim = x_dim
        self.r_dim = r_dim
        self.h_dim = h_dim
        self.uses_r = uses_r
        
        # Store noise levels as tensor for broadcasting
        self.register_buffer('x_std_per_dim', 
                            torch.tensor(x_std_per_dim, dtype=torch.float32))
        
        # Input: x_noisy + (optionally) r + noise_levels
        input_dim = x_dim
        if uses_r:
            input_dim += r_dim
        input_dim += x_dim  # Add noise levels as features
        
        # Network outputs: mean and log_var for each dimension
        self.net = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim * 2)  # mu and log_var for each dim
        )
    
    def forward(self, x_noisy, r=None):
        """
        Parameters
        ----------
        x_noisy : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        r : torch.Tensor or None
            Shape (batch_size, r_dim) if uses_r=True
        
        Returns
        -------
        q_x_star : Normal distribution
        x_star_mu : torch.Tensor
        x_star_sigma : torch.Tensor
        """
        batch_size, num_points, x_dim = x_noisy.shape
        
        # Prepare noise level features (broadcast to match x_noisy shape)
        noise_features = self.x_std_per_dim.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_points, x_dim
        )
        
        # Concatenate inputs
        inputs = [x_noisy, noise_features]
        
        if self.uses_r and r is not None:
            # Expand r to match num_points
            r_expanded = r.unsqueeze(1).expand(batch_size, num_points, self.r_dim)
            inputs.append(r_expanded)
        
        net_input = torch.cat(inputs, dim=-1)
        
        # Flatten for network
        net_input_flat = net_input.view(batch_size * num_points, -1)
        
        # Forward
        output = self.net(net_input_flat)
        output = output.view(batch_size, num_points, x_dim * 2)
        
        # Split into mu and log_var
        x_star_mu = output[:, :, :x_dim]
        x_star_log_var = output[:, :, x_dim:]
        x_star_sigma = 0.1 + 0.9 * torch.sigmoid(x_star_log_var)
        
        # Create distribution
        q_x_star = Normal(x_star_mu, x_star_sigma)
        
        return q_x_star, x_star_mu, x_star_sigma


class Encoder(nn.Module):
    """Encoder: (x, y) → r_i"""
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, r_dim)
        )
    
    def forward(self, x, y):
        """
        x: (batch_size, x_dim) or (batch*num_points, x_dim)
        y: (batch_size, y_dim) or (batch*num_points, y_dim)
        """
        return self.net(torch.cat([x, y], dim=-1))


class MuSigmaEncoder(nn.Module):
    """Maps r → (mu_z, sigma_z)"""
    def __init__(self, r_dim, z_dim):
        super().__init__()
        self.r_dim = r_dim
        self.z_dim = z_dim
        
        self.net = nn.Sequential(
            nn.Linear(r_dim, r_dim),
            nn.ReLU(),
            nn.Linear(r_dim, z_dim * 2)
        )
    
    def forward(self, r):
        """
        r: (batch_size, r_dim)
        Returns: mu, sigma
        """
        output = self.net(r)
        mu = output[:, :self.z_dim]
        log_var = output[:, self.z_dim:]
        sigma = 0.1 + 0.9 * torch.sigmoid(log_var)
        return mu, sigma


class Decoder(nn.Module):
    """Decoder: (x, z) → (mu_y, sigma_y)"""
    def __init__(self, x_dim, z_dim, h_dim, y_dim):
        super().__init__()
        self.y_dim = y_dim
        
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, y_dim * 2)
        )
    
    def forward(self, x, z):
        """
        x: (batch_size, num_points, x_dim)
        z: (batch_size, z_dim)
        """
        batch_size, num_points, _ = x.shape
        
        # Expand z to match points
        z_expanded = z.unsqueeze(1).expand(batch_size, num_points, -1)
        
        # Concatenate and flatten
        inputs = torch.cat([x, z_expanded], dim=-1)
        inputs_flat = inputs.view(batch_size * num_points, -1)
        
        # Forward
        output = self.net(inputs_flat)
        output = output.view(batch_size, num_points, self.y_dim * 2)
        
        mu_y = output[:, :, :self.y_dim]
        log_var_y = output[:, :, self.y_dim:]
        sigma_y = 0.1 + 0.9 * F.softplus(log_var_y)
        
        return mu_y, sigma_y


class DenoisingNeuralProcessMultiNoise(nn.Module):
    """
    Neural Process with coordinate denoising.
    Supports different noise levels per dimension.
    """
    
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim, 
                 x_std_per_dim=None, denoise_uses_y=True):
        """
        Parameters
        ----------
        x_dim : int
            Total dimension of input coordinates
        y_dim : int
            Dimension of outputs
        r_dim, z_dim, h_dim : int
            Network dimensions
        x_std_per_dim : list or dict
            Noise level for each dimension
            - If list: [σ_0, σ_1, σ_2, ...]
            - If dict: {0: σ_0, 2: σ_2, ...} (missing dims = 0)
        denoise_uses_y : bool
            Whether denoiser uses y information
        """
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.denoise_uses_y = denoise_uses_y
        
        #  Convert to list of noise levels
        if x_std_per_dim is None:
            self.x_std_per_dim = [0.1] * x_dim
        elif isinstance(x_std_per_dim, dict):
            self.x_std_per_dim = [x_std_per_dim.get(i, 0.0) for i in range(x_dim)]
        else:
            self.x_std_per_dim = list(x_std_per_dim)
        
        # Denoiser with per-dim noise
        self.x_to_x_star = XStarEncoderMultiNoise(
            x_dim=x_dim,
            r_dim=r_dim if denoise_uses_y else 0,
            h_dim=h_dim,
            x_std_per_dim=self.x_std_per_dim,
            uses_r=denoise_uses_y
        )
        
        # Two encoders
        self.encoder_noisy = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.encoder_clean = Encoder(x_dim, y_dim, h_dim, r_dim)
        
        # Latent and decoder
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)
    
    def aggregate(self, r_i):
        """Aggregate representations: (batch, points, r_dim) → (batch, r_dim)"""
        return r_i.mean(dim=1)
    
    def encode_context(self, x, y, encoder):
        """Encode and aggregate context"""
        batch_size, num_points, _ = x.size()
        
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        
        r_i_flat = encoder(x_flat, y_flat)
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        r = self.aggregate(r_i)
        
        return r
    
    def xy_to_mu_sigma(self, x, y, encoder):
        """Encode to latent parameters"""
        r = self.encode_context(x, y, encoder)
        mu, sigma = self.r_to_mu_sigma(r)
        return mu, sigma
    
    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Forward pass with per-dimension noise.
        
        Returns
        -------
        If y_target provided: (p_y_pred, q_z_tgt, q_z_ctx, q_xs_ctx)
        If y_target is None: p_y_pred only
        """
        
        # 1) Optionally encode noisy context
        if self.denoise_uses_y:
            r_noisy = self.encode_context(x_context, y_context, self.encoder_noisy)
        else:
            r_noisy = None
        
        # 2) Denoise context coordinates
        q_xs_ctx, _, _ = self.x_to_x_star(x_context, r_noisy)
        xstar_sample_ctx = q_xs_ctx.rsample()
        
        # 3) Re-encode with denoised context
        r_clean = self.encode_context(xstar_sample_ctx, y_context, self.encoder_clean)
        
        # 4) Target coords are clean (no denoising)
        xstar_target = x_target
        
        # 5) Context posterior q(z | x*_ctx, y_ctx)
        mu_z_ctx, sig_z_ctx = self.xy_to_mu_sigma(
            xstar_sample_ctx, y_context, self.encoder_clean
        )
        q_z_ctx = Normal(mu_z_ctx, sig_z_ctx)
        
        if y_target is not None:
            # Training mode: full outputs
            mu_z_tgt, sig_z_tgt = self.xy_to_mu_sigma(
                xstar_target, y_target, self.encoder_clean
            )
            q_z_tgt = Normal(mu_z_tgt, sig_z_tgt)
            
            z_sample = q_z_tgt.rsample()
            y_mu, y_sig = self.xz_to_y(xstar_target, z_sample)
            p_y_pred = Normal(y_mu, y_sig)
            
            return p_y_pred, q_z_tgt, q_z_ctx, q_xs_ctx
        else:
            # Prediction mode
            z_sample = q_z_ctx.rsample()
            y_mu, y_sig = self.xz_to_y(xstar_target, z_sample)
            p_y_pred = Normal(y_mu, y_sig)
            
            return p_y_pred
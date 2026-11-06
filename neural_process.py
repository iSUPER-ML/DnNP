import torch
from models import Encoder, MuSigmaEncoder, Decoder, XStarEncoder
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import torch.nn.functional as F

class NeuralProcess(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim):
        super(NeuralProcess, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        # Initialize networks
        self.xy_to_r = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)



    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """

        mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
        # Sample from distribution based on context
        q_context = Normal(mu_context, sigma_context)
        if self.training:
            # Encode target (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            z_sample = q_target.rsample()
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:

            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred

            
class NeuralProcessImg(nn.Module):
    """
    Wraps regular Neural Process for image processing.

    Parameters
    ----------
    img_size : tuple of ints
        E.g. (1, 28, 28) or (3, 32, 32)

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, img_size, r_dim, z_dim, h_dim):
        super(NeuralProcessImg, self).__init__()
        self.img_size = img_size
        self.num_channels, self.height, self.width = img_size
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.neural_process = NeuralProcess(x_dim=2, y_dim=self.num_channels,
                                            r_dim=r_dim, z_dim=z_dim,
                                            h_dim=h_dim)

    def forward(self, img, context_mask, target_mask):
        """
        Given an image and masks of context and target points, returns a
        distribution over pixel intensities at the target points.

        Parameters
        ----------
        img : torch.Tensor
            Shape (batch_size, channels, height, width)

        context_mask : torch.ByteTensor
            Shape (batch_size, height, width). Binary mask indicating
            the pixels to be used as context.

        target_mask : torch.ByteTensor
            Shape (batch_size, height, width). Binary mask indicating
            the pixels to be used as target.
        """
        x_context, y_context = img_mask_to_np_input(img, context_mask)
        x_target, y_target = img_mask_to_np_input(img, target_mask)
        return self.neural_process(x_context, y_context, x_target, y_target)


class DenoisingNeuralProcess(nn.Module):
    """
    Implements Neural Process with coordinate denoising.
    
    Parameters
    ----------
    x_dim : int
        Dimension of x values (coordinates).
    y_dim : int
        Dimension of y values (observations).
    r_dim : int
        Dimension of representation r.
    z_dim : int
        Dimension of latent variable z.
    h_dim : int
        Dimension of hidden layers.
    x_std : float or None
        Noise standard deviation for coordinates.
        If None, sigma is learned as a global parameter.
    denoise_uses_y : bool
        If True, use y information for denoising (current behavior).
        If False, denoise using only x (theory-pure).
    """
    
    def __init__(self, x_dim, y_dim, r_dim, z_dim, h_dim, x_std=None, denoise_uses_y=True):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.denoise_uses_y = denoise_uses_y

        # --- Sigma handling ---
        # If x_std is None, learn log_sigma; otherwise fix it as a buffer.
        if x_std is None:
            self.log_sigma = nn.Parameter(torch.tensor(0.0))  # init: σ ≈ 0.69 via softplus
            self._fixed_sigma = None
        else:
            self.log_sigma = None
            self._fixed_sigma = x_std

        # Denoising network — x_std no longer passed at init; sigma
        # is injected dynamically at forward time via x_to_x_star.forward().
        # NOTE: XStarEncoder.forward() must accept an x_std kwarg for this.
        self.x_to_x_star = XStarEncoder(
            x_dim=x_dim,
            r_dim=r_dim if denoise_uses_y else 0,
            h_dim=h_dim,
            uses_r=denoise_uses_y
        )
        
        # Two encoders
        self.encoder_noisy = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.encoder_clean = Encoder(x_dim, y_dim, h_dim, r_dim)
        
        # Latent and decoder
        self.r_to_mu_sigma = MuSigmaEncoder(r_dim, z_dim)
        self.xz_to_y = Decoder(x_dim, z_dim, h_dim, y_dim)

    @property
    def sigma(self):
        """Returns current noise std — learned or fixed."""
        if self._fixed_sigma is not None:
            return self._fixed_sigma
        return F.softplus(self.log_sigma)  # always positive
    
    def aggregate(self, r_i):
        return torch.mean(r_i, dim=1)
    
    def encode_context(self, x, y, encoder):
        batch_size, num_points, _ = x.size()
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        r_i_flat = encoder(x_flat, y_flat)
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        r = self.aggregate(r_i)
        return r
    
    def xy_to_mu_sigma(self, x, y, encoder):
        r = self.encode_context(x, y, encoder)
        mu, sigma = self.r_to_mu_sigma(r)
        return mu, sigma
    
    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Forward pass with coordinate denoising.
        
        Parameters
        ----------
        x_context : torch.Tensor
            Noisy context coordinates [batch, num_context, x_dim]
        y_context : torch.Tensor
            Context values [batch, num_context, y_dim]
        x_target : torch.Tensor
            Clean target coordinates [batch, num_target, x_dim]
        y_target : torch.Tensor, optional
            Target values (if provided, returns full outputs for loss computation)
        
        Returns
        -------
        If y_target provided: (p_y_pred, q_z_tgt, q_z_ctx, q_xs_ctx)
        If y_target is None: p_y_pred only
        """
        
        # ------- 1) Optionally encode noisy context -------
        if self.denoise_uses_y:
            r_noisy = self.encode_context(x_context, y_context, self.encoder_noisy)
        else:
            r_noisy = None
        
        # ------- 2) Denoise context coordinates -------
        # Pass self.sigma dynamically so gradients flow when sigma is learned.
        if self.denoise_uses_y:
            q_xs_ctx, _, _ = self.x_to_x_star(x_context, r_noisy, x_std=self.sigma)
        else:
            q_xs_ctx, _, _ = self.x_to_x_star(x_context, None, x_std=self.sigma)
        
        # Sample denoised coordinates
        xstar_sample_ctx = q_xs_ctx.rsample()
        
        # ------- 3) Re-encode with denoised context -------
        r_clean = self.encode_context(xstar_sample_ctx, y_context, self.encoder_clean)
        
        # ------- 4) Target coords are clean (no denoising) -------
        xstar_target = x_target
        
        # ------- 5) Context posterior q(z | x*_ctx, y_ctx) -------
        mu_z_ctx, sig_z_ctx = self.xy_to_mu_sigma(
            xstar_sample_ctx, y_context, self.encoder_clean
        )
        q_z_ctx = Normal(mu_z_ctx, sig_z_ctx)

        if y_target is not None:
            mu_z_tgt, sig_z_tgt = self.xy_to_mu_sigma(
                xstar_target, y_target, self.encoder_clean
            )
            q_z_tgt = Normal(mu_z_tgt, sig_z_tgt)
            z_sample = q_z_tgt.rsample()
            y_mu, y_sig = self.xz_to_y(xstar_target, z_sample)
            p_y_pred = Normal(y_mu, y_sig)
            return p_y_pred, q_z_tgt, q_z_ctx, q_xs_ctx
            
        else:
            z_sample = q_z_ctx.rsample()
            y_mu, y_sig = self.xz_to_y(xstar_target, z_sample)
            p_y_pred = Normal(y_mu, y_sig)
            return p_y_pred


class DenoisingNeuralProcessImg(nn.Module):
    """
    Wrapper for image tasks.
    """
    
    def __init__(self, img_size, r_dim, z_dim, h_dim, x_std=0.1, 
                 dropout=0.0, denoise_depth=3, denoise_width=256,
                 denoise_uses_y=True):
        super().__init__()
        self.img_size = img_size
        self.num_channels, self.height, self.width = img_size
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_std = x_std
        
        self.neural_process = DenoisingNeuralProcess(
            x_dim=2,  # (row, col) coordinates
            y_dim=self.num_channels,
            r_dim=r_dim,
            z_dim=z_dim,
            h_dim=h_dim,
            x_std=x_std,
            denoise_uses_y=denoise_uses_y  
        )
    
    def forward(self, x_context, y_context, x_target, y_target=None):
        """Forward pass."""
        return self.neural_process(x_context, y_context, x_target, y_target)



class NPregression(nn.Module):
    def __init__(self, model_name, dim_input, dim_output, dim_hidden=128, num_heads=8):
        super(NPregression, self).__init__()
        self.model_name = model_name
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

        dec_dim = dim_input + dim_hidden * 2
            
        self.x_MLP = nn.Sequential(nn.Linear(dim_input, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden))
        
        if 'LD' in self.model_name:
            r_dim = dim_hidden * dim_output * 2
            z_dim = dim_hidden * dim_output * 2
            
            self.param_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                           nn.ReLU(),
                                           nn.Linear(dim_hidden, dim_hidden))
        
            self.layernorm = nn.LayerNorm(dim_hidden)
            
        else:
            r_dim = dim_hidden
            z_dim = dim_hidden
            self.decoding = nn.Sequential(nn.Linear(dec_dim, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_hidden),
                                          nn.ReLU(),
                                          nn.Linear(dim_hidden, dim_output * 2))
    
        
        if 'ANP' in self.model_name:
            self.e_MLP = nn.Sequential(nn.Linear(dim_input, dim_hidden),
                                       nn.ReLU(),
                                       nn.Linear(dim_hidden, r_dim))
        
            self.cross_Attention = nn.MultiheadAttention(r_dim, num_heads)
        
            self.r_Attention = nn.MultiheadAttention(r_dim, num_heads)
            
            self.sub_Layer = nn.Sequential(nn.Linear(r_dim, r_dim * 2),
                                           nn.ReLU(),
                                           nn.Linear(r_dim * 2, r_dim))
            
            self.layer_Norm = nn.LayerNorm(r_dim)
        
            self.s_Attention = nn.MultiheadAttention(dim_hidden, num_heads)
        
        self.r_MLP = nn.Sequential(nn.Linear(dim_hidden + dim_output, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, r_dim))
        
        self.s_MLP = nn.Sequential(nn.Linear(dim_hidden + dim_output, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, dim_hidden))
            
        self.z_MLP = nn.Sequential(nn.Linear(dim_hidden, dim_hidden),
                                   nn.ReLU(),
                                   nn.Linear(dim_hidden, z_dim * 2))
        
        
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
    
    def deterministic_encoder(self, det_enc_inp, x_context=None, x_target=None, attention=False, toy_attention=True):
        if attention == False:
            ri = self.r_MLP(det_enc_inp)
            r = ri.mean(1, True)
        else:
            if toy_attention:
                _, num_context, _ = det_enc_inp.size()
                query = self.e_MLP(x_target).permute(1,0,2)
                key = self.e_MLP(x_context).permute(1,0,2)
                value = self.r_MLP(det_enc_inp).permute(1,0,2)
                out, _ = self.cross_Attention(query, key, value)
                r = out.permute(1,0,2).unsqueeze(2)
            else:
                hidden1 = self.r_MLP(det_enc_inp).permute(1,0,2)
                hidden2, _ = self.r_Attention(hidden1, hidden1, hidden1)
                ri, _ = self.r_Attention(hidden2, hidden2, hidden2)
                
                query = self.e_MLP(x_target).permute(1,0,2)
                key = self.e_MLP(x_context).permute(1,0,2)
                out, _ = self.cross_Attention(query, key, ri)
                sub1 = out + query
                sub2 = self.layer_Norm(sub1.permute(1,0,2))
                r = self.layer_Norm(self.sub_Layer(sub2)+sub2).unsqueeze(2)
        return r        
        
    def reparameterize(self, dist):
        eps = torch.randn(size = dist.stddev.size()).to(self.device)
        sample = dist.mean + dist.stddev * eps
        return sample
    
    def stochastic_distribution(self, z_param):
        half = int(z_param.size(-1)/2)
        z_mu = z_param[:, :, :half]
        z_omega = z_param[:, :, half:]
        z_sigma = 0.1+0.9*self.sigmoid(z_omega)
        z_dist = torch.distributions.normal.Normal(z_mu, z_sigma)
        return z_dist
    
    def stochastic_encoder(self, lat_enc_inp, attention=False):
        if attention == False:
            si = self.s_MLP(lat_enc_inp)
            s = si.mean(1, True)
            z_param = self.z_MLP(s)
        else:
            hidden1 = self.s_MLP(lat_enc_inp).permute(1,0,2)
            hidden2, _ = self.s_Attention(hidden1, hidden1, hidden1)
            si, _ = self.s_Attention(hidden2, hidden2, hidden2)
            s = si.permute(1,0,2).mean(1, True)
            z_param = self.z_MLP(s)
        z_dist = self.stochastic_distribution(z_param)
        return z_dist
    
    def decoder(self, x, r, z, linear):
        if linear == False:
            if r.dim() == 4 and r.size(2) == 1:
                r = r.squeeze(2)
            if z.dim() == 4 and z.size(2) == 1:
                z = z.squeeze(2)

            dec_inp = torch.cat((x, r, z), 2)
            param = self.decoding(dec_inp)
            mu = param[:, :, :self.dim_output]
            omega = param[:, :, self.dim_output:]
            sigma = 0.1+0.9*self.softplus(omega)
            dist = torch.distributions.normal.Normal(mu, sigma)
        else:
            r = r.reshape(-1, r.size(1), self.dim_output * 2, self.dim_hidden)
            z = z.reshape(-1, z.size(1), self.dim_output * 2, self.dim_hidden)
            dec_param = r + self.param_MLP(z)
            dec_param = self.layernorm(dec_param).permute(0,1,3,2)
            
            param = torch.matmul(self.x_MLP(x).unsqueeze(2), dec_param).squeeze()
            mu = param[:,:,:self.dim_output]
            omega = param[:,:,self.dim_output:]
            sigma = 0.1+0.9*self.softplus(omega)
            dist = torch.distributions.normal.Normal(mu, sigma)
        return dist

    def forward(self, x_context, y_context, x_target, y_target=None, toy_attention=True):
        _, num_context, _ = x_context.size()
        _, num_target, _ = x_target.size()

        context = torch.cat((self.x_MLP(x_context), y_context), 2)
        linear = 'LD' in self.model_name

        if 'CNP' in self.model_name:
            r_c = self.deterministic_encoder(context)
            r_c = r_c.unsqueeze(1).repeat(1, num_target, 1, 1)
            batch_size = x_target.size(0)
            z = torch.zeros(batch_size, num_target, 1, self.dim_hidden).to(self.device)
            KL = torch.tensor([0.0]).to(self.device)

        elif 'ANP' in self.model_name:
            r_c = self.deterministic_encoder(context, x_context, x_target,
                                            attention=True, toy_attention=toy_attention)
            z_prior = self.stochastic_encoder(context, attention=True)
            if y_target is not None:
                target = torch.cat((self.x_MLP(x_target), y_target), 2)
                z_posterior = self.stochastic_encoder(target, attention=True)
                z_sample = self.reparameterize(z_posterior)
                KL = torch.distributions.kl.kl_divergence(z_posterior, z_prior).sum(-1).mean()
            else:
                z_sample = z_prior.rsample()
                KL = torch.tensor([0.0]).to(self.device)
            z = z_sample.unsqueeze(1).repeat(1, num_target, 1, 1)

        elif 'NP' in self.model_name:
            r_c = self.deterministic_encoder(context)
            r_c = r_c.unsqueeze(1).repeat(1, num_target, 1, 1)
            z_prior = self.stochastic_encoder(context)
            if y_target is not None:
                target = torch.cat((self.x_MLP(x_target), y_target), 2)
                z_posterior = self.stochastic_encoder(target)
                z_sample = self.reparameterize(z_posterior)
                KL = torch.distributions.kl.kl_divergence(z_posterior, z_prior).sum(-1).mean()
            else:
                z_sample = z_prior.rsample()   # prediction mode: sample from context prior
                KL = torch.tensor([0.0]).to(self.device)
            z = z_sample.unsqueeze(1).repeat(1, num_target, 1, 1)

        y_dist = self.decoder(x_target, r_c, z, linear=linear)

        if y_target is not None:
            NLL = -y_dist.log_prob(y_target).sum(-1).mean()
            MSE = ((y_target - y_dist.mean) ** 2).sum(-1).mean()
            return y_dist, KL, NLL, MSE

        return y_dist   # prediction mode
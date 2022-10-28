"""
Success Prediction Module
"""

import mrl
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from mrl.replays.online_her_buffer import OnlineHERBuffer

def relative_probs_from_log_probs(log_probs):
    """
    Returns relative probability from the log probabilities. They're not exactly
    equal to the probability, but relative scalings between them are all maintained.

    For correctness, all log_probs must be passed in at the same time.
    """
    probs = np.exp(log_probs - log_probs.mean())
    probs = np.clip(probs, 0, 1e6)
    print(np.max(probs), np.min(probs))
    return probs

def compute_log_p_log_q_log_d(
    model,
    data,
    num_latents_to_sample=1,
    device="cuda:0"
):
    data = model.normalize(data)
    data = torch.from_numpy(data).to(device)
    latent_distribution_params = model.encoder(data)
    batch_size = data.shape[0]
    representation_size = model.latent_dim
    log_p, log_q, log_d = torch.zeros((batch_size, num_latents_to_sample)), torch.zeros(
        (batch_size, num_latents_to_sample)), torch.zeros((batch_size, num_latents_to_sample))
    true_prior = Normal(torch.zeros((batch_size, representation_size), device="cuda:0"),
                        torch.ones((batch_size, representation_size), device="cuda:0"))
    mus = latent_distribution_params[..., :representation_size]
    logvars = latent_distribution_params[..., representation_size:]
    for i in range(num_latents_to_sample):
        latents = model.rsample(mus, logvars)
        stds = logvars.exp().pow(.5)
        vae_dist = Normal(mus, stds)
        log_p_z = true_prior.log_prob(latents).sum(dim=1)
        log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=1)
        dec_mu = model.decoder(latents)
        dec_var = torch.ones_like(dec_mu)
        decoder_dist = Normal(dec_mu, dec_var.pow(.5))
        # no needs to denormalize here.
        log_d_x_given_z = decoder_dist.log_prob(data).sum(dim=1)

        log_p[:, i] = log_p_z.detach().cpu()
        log_q[:, i] = log_q_z_given_x.detach().cpu()
        log_d[:, i] = log_d_x_given_z.detach().cpu()
    return log_p, log_q, log_d


def compute_p_x_np_to_np(
    model,
    data,
    power,
    num_latents_to_sample=10,
    device="cuda:0"
):

    log_p, log_q, log_d = compute_log_p_log_q_log_d(
        model,
        data,
        num_latents_to_sample,
        device
    )

    log_p_x = (log_p - log_q + log_d).mean(dim=1) # normalized.

    log_p_x_skewed = power * log_p_x
    return log_p_x_skewed


class VAEPredictor(mrl.Module):
  """Predicts success using a learned discriminator"""

  def __init__(self, batch_size = 64, history_length = 50000, optimize_every=4000, log_every=4000):
    # 50 means 750 for previous data. 2500 for ant.
    super().__init__(
      'vae_predictor',
      required_agent_modules=[
        'env', 'replay_buffer', 'vae'
      ],
      locals=locals())
    self.log_every = log_every
    self.batch_size = batch_size
    self.history_length = history_length
    self.optimize_every = optimize_every
    self.opt_steps = 0
    self.sample_log_priorities = np.zeros(self.history_length, dtype=np.float32)
    self.sample_priorities = np.zeros(self.history_length, dtype=np.float32)
    self.update_priority_steps = 2000
    self.beta = 10
    self.ready = False
    self.recent_samples = np.array([])

  def _setup(self):
    super()._setup()
    assert isinstance(self.replay_buffer, OnlineHERBuffer)
    assert self.env.goal_env
    self.n_envs = self.env.num_envs
    self.optimizer = torch.optim.Adam(self.vae.model.parameters(), lr=1e-3)
    self.opt_freq = self.config.optimize_every
    self.vae_norm_input = self.config.other_args["vae_norm_input"]
    self.optimize_every = self.optimize_every // self.opt_freq

  def update_priorities(self, recent_samples):
      for i in range(0, len(recent_samples), self.update_priority_steps):
          samples = recent_samples[i:i+self.update_priority_steps]
          self.sample_log_priorities[i:i+len(samples)] = compute_p_x_np_to_np(self.vae.model, samples, -2.5, 10, self.config.device).numpy()
          #self.sample_log_priorities[i:i+len(samples)] = compute_p_x_np_to_np(self.vae.model, samples, -5, 10, self.config.device).numpy()
      self.sample_priorities[:len(recent_samples)] = relative_probs_from_log_probs(self.sample_log_priorities[:len(recent_samples)])
      self.sample_priorities[:len(recent_samples)] = self.sample_priorities[:len(recent_samples)] / np.sum(self.sample_priorities[:len(recent_samples)])

  def sample_ags(self, n, generation=True):
    if not generation and len(self.recent_samples) > 0:
      num_samples = len(self.recent_samples)
      sample_ids = np.random.choice(num_samples, n, p=self.sample_priorities[:num_samples])
      decoded_ags = self.recent_samples[sample_ids]
    else:
      sampled_latents = torch.randn(n, self.vae.model.latent_dim, device=self.config.device)
      decoded_ags = self.vae.model.decoder(sampled_latents)
      decoded_ags = self.vae.model.denormalize(decoded_ags.detach().cpu().numpy())
    return decoded_ags

  def _optimize(self):
    self.opt_steps += 1
    # 40000 follows the settings described in the original paper of Skew-Fit.
    if self.opt_steps < 40000//self.opt_freq:
        num_updates = 1000 # 1000 -> train on 100000
    else:
        num_updates = 200

    if len(self.replay_buffer) > self.batch_size and self.opt_steps % self.optimize_every == 0:
      losses = []
      #recent_samples_indices = np.random.choice(len(self.replay_buffer), self.history_length)
      recent_samples_indices = np.arange(max(0, len(self.replay_buffer) - self.history_length), len(self.replay_buffer))
      recent_samples = self.replay_buffer.buffer.BUFF.buffer_ag.get_batch(recent_samples_indices)
      num_samples = len(recent_samples_indices)
      if self.opt_steps >= 40000//self.opt_freq:
          self.update_priorities(recent_samples)
          self.recent_samples = recent_samples

      if self.vae_norm_input:
        self.vae.model.get_normalize_parameters(recent_samples)
          
      for i in range(num_updates):
          if self.opt_steps <  40000//self.opt_freq:
            batch_indices = np.random.randint(num_samples, size=self.batch_size)
          else:
            batch_indices_priority = np.random.choice(num_samples, self.batch_size, p=self.sample_priorities[:num_samples])            
            batch_indices_random = np.random.randint(num_samples, size=self.batch_size)
            batch_indices = np.concatenate([batch_indices_priority, batch_indices_random])

          batch_ags = self.vae.model.normalize(recent_samples[batch_indices])
          batch_ags = torch.from_numpy(batch_ags).to(self.config.device)
          reconstruction, latent_dist = self.vae(batch_ags)
          log_prob = self.vae.model.log_prob(batch_ags, reconstruction)
          kle = self.vae.model.kl_divergence(latent_dist[0], latent_dist[1])
          loss = - log_prob + kle * self.beta
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          # no needs to denormalize here as we only want the loss..
          losses.append(loss.detach().cpu().numpy())
      self.ready = True

      if hasattr(self, 'logger'):
        self.logger.add_histogram('vae_loss', np.mean(losses), self.log_every)
      print(self.opt_steps, "Update the vae_predictor", np.mean(losses), self.log_every)


  def save(self, save_folder : str):
    pass

  def load(self, save_folder : str):
    pass


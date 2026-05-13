"""
Neural LDPC min-sum decoder (PyTorch).

This module implements a trainable/min-sum style message-passing decoder
with learnable scaling factors per iteration.
"""
from typing import Optional

import torch
import torch.nn as nn


class NeuralMinSumDecoder(nn.Module):
	"""Neural Min-Sum LDPC decoder.

	Args:
		H: parity-check matrix (torch.Tensor or numpy array) of shape (m, n)
		num_iterations: number of message-passing iterations
		learnable_scaling: if True, learn a positive scale per iteration
		learnable_beta: if True, learn a non-negative offset (beta) per iteration for offset min-sum
	"""

	def __init__(self, H, num_iterations: int = 5, learnable_scaling: bool = True, learnable_beta: bool = False):
		super().__init__()
		H = torch.as_tensor(H)
		assert H.dim() == 2, "H must be a 2D parity-check matrix"
		self.register_buffer("H", H.float())
		self.m, self.n = H.shape
		self.num_iterations = int(num_iterations)

		# Precompute neighbor indices
		# check_to_vars: list of variable indices for each check
		# var_to_checks: list of check indices for each variable
		self.check_to_vars = [torch.nonzero(self.H[i], as_tuple=False).view(-1) for i in range(self.m)]
		self.var_to_checks = [torch.nonzero(self.H[:, j], as_tuple=False).view(-1) for j in range(self.n)]

		# Initialize alpha so that sigmoid(raw_alpha) = 0.7 by default
		# Keep a separate alpha per iteration and per check node: shape (num_iterations, m)
		alpha_init_val = 0.7
		# inverse sigmoid (logit), clamp to avoid 0/1 extremes
		a_v = torch.tensor(alpha_init_val).clamp(1e-6, 1 - 1e-6)
		alpha_inv = torch.log(a_v / (1.0 - a_v))
		raw_alpha_init = alpha_inv * torch.ones((self.num_iterations, self.m))
		if learnable_scaling:
			# positive scalar per (iteration, check)
			self.raw_alpha = nn.Parameter(raw_alpha_init)
		else:
			self.register_buffer("raw_alpha", raw_alpha_init)

		# beta (offset) per iteration and per check for offset min-sum.
		# Constrain beta to [0,1) via sigmoid mapping. Initialize near 0 by default.
		beta_init_val = 0.0
		b_v = torch.tensor(beta_init_val).clamp(0.0 + 1e-6, 1.0 - 1e-6)
		beta_inv = torch.log(b_v / (1.0 - b_v))
		raw_beta_init = beta_inv * torch.ones((self.num_iterations, self.m))
		if learnable_beta:
			self.raw_beta = nn.Parameter(raw_beta_init)
		else:
			self.register_buffer("raw_beta", raw_beta_init)

	def alpha(self):
		# return scaling factors constrained to (0,1]
		# use sigmoid mapping to keep values in (0,1). Clamp a tiny eps to ensure strict positivity.
		eps = 1e-6
		return torch.sigmoid(self.raw_alpha).clamp(min=eps, max=1.0)

	def beta(self):
		# return offset factors constrained to [0,1)
		# use sigmoid mapping to keep values in (0,1). Clamp max below 1.0 to enforce <1.
		eps = 1e-6
		return torch.sigmoid(self.raw_beta).clamp(min=0.0, max=1.0 - eps)

	def set_alpha_trainable(self, trainable: bool = True):
		"""Enable or disable training of the raw alpha parameters at runtime.

		When enabling, any existing buffer is converted to an nn.Parameter so it
		appears in model.parameters(); when disabling, the parameter is converted
		back to a buffer (non-trainable).
		"""
		# convert buffer -> parameter
		if trainable:
			if "raw_alpha" in self._buffers:
				raw = self._buffers.pop("raw_alpha")
				# register as parameter
				self.register_parameter("raw_alpha", nn.Parameter(raw))
			# if already a parameter, nothing to do
		else:
			# convert parameter -> buffer
			if "raw_alpha" in self._parameters:
				param = self._parameters.pop("raw_alpha")
				# register as buffer (detached)
				self.register_buffer("raw_alpha", param.detach())

	def set_beta_trainable(self, trainable: bool = True):
		"""Enable or disable training of the raw beta parameters at runtime.

		Similar semantics to set_alpha_trainable.
		"""
		if trainable:
			if "raw_beta" in self._buffers:
				raw = self._buffers.pop("raw_beta")
				self.register_parameter("raw_beta", nn.Parameter(raw))
		else:
			if "raw_beta" in self._parameters:
				param = self._parameters.pop("raw_beta")
				self.register_buffer("raw_beta", param.detach())

	def set_alpha_init(self, values):
		"""Initialize raw_alpha so sigmoid(raw_alpha)=values.

		values: scalar or iterable of length num_iterations
		"""
		v = torch.as_tensor(values, dtype=self.raw_alpha.dtype, device=self.raw_alpha.device)
		# Accept scalar, (num_iterations,), or (num_iterations, m)
		if v.numel() == 1:
			v = v.repeat(self.num_iterations, self.m)
		elif v.numel() == self.num_iterations:
			v = v.view(self.num_iterations, 1).repeat(1, self.m)
		elif v.numel() == self.num_iterations * self.m:
			v = v.view(self.num_iterations, self.m)
		else:
			raise ValueError("values must be scalar, length num_iterations, or shape (num_iterations, m)")
		# inverse sigmoid (logit): raw = log(v/(1-v)). Clamp v into (0,1) first.
		v = v.clamp(1e-6, 1.0 - 1e-6)
		inv = torch.log(v / (1.0 - v))
		if "raw_alpha" in self._parameters:
			with torch.no_grad():
				self.raw_alpha.copy_(inv)
		else:
			with torch.no_grad():
				self._buffers["raw_alpha"].copy_(inv)

	def set_beta_init(self, values):
		"""Initialize raw_beta so sigmoid(raw_beta)=values.

		values: scalar or iterable of length num_iterations
		"""
		v = torch.as_tensor(values, dtype=self.raw_beta.dtype, device=self.raw_beta.device)
		# Accept scalar, (num_iterations,), or (num_iterations, m)
		if v.numel() == 1:
			v = v.repeat(self.num_iterations, self.m)
		elif v.numel() == self.num_iterations:
			v = v.view(self.num_iterations, 1).repeat(1, self.m)
		elif v.numel() == self.num_iterations * self.m:
			v = v.view(self.num_iterations, self.m)
		else:
			raise ValueError("values must be scalar, length num_iterations, or shape (num_iterations, m)")
		# inverse sigmoid (logit): raw = log(v/(1-v)). Clamp v into (0,1) first.
		v = v.clamp(1e-6, 1.0 - 1e-6)
		inv = torch.log(v / (1.0 - v))
		if "raw_beta" in self._parameters:
			with torch.no_grad():
				self.raw_beta.copy_(inv)
		else:
			with torch.no_grad():
				self._buffers["raw_beta"].copy_(inv)

	def forward(self, llr: torch.Tensor, return_llr: bool = False):
		"""Run neural min-sum decoding.

		Args:
			llr: tensor of shape (batch, n) or (n,) representing input log-likelihood ratios
			return_llr: if True, return final posterior LLRs in addition to hard bits

		Returns:
			decoded_bits: tensor of shape (batch, n) (or (n,)) with values 0/1
			optionally final_llr
		"""
		single = False
		if llr.dim() == 1:
			llr = llr.unsqueeze(0)
			single = True

		batch_size = llr.shape[0]
		device = llr.device
		dtype = llr.dtype

		# Move precomputed indices to device
		check_to_vars = [idx.to(device) for idx in self.check_to_vars]
		var_to_checks = [idx.to(device) for idx in self.var_to_checks]

		# Initialize messages: v2c (check x var) with zeros where H=1
		v2c = torch.zeros((self.m, self.n, batch_size), device=device, dtype=dtype)
		mask = (self.H > 0).to(device=device)

		# Initialize v2c with channel llr: for an edge (i,j), initial v2c = llr_j
		for j in range(self.n):
			checks = var_to_checks[j]
			if checks.numel() == 0:
				continue
			v2c[checks, j, :] = llr[:, j].transpose(0, 0)

		alphas = self.alpha().to(device)
		betas = self.beta().to(device)

		for it in range(self.num_iterations):
			# Check-node update: compute c2v
			c2v = torch.zeros_like(v2c)

			for i in range(self.m):
				vars_idx = check_to_vars[i]
				if vars_idx.numel() == 0:
					continue
				msgs = v2c[i, vars_idx, :]  # shape (d, batch)
				signs = torch.sign(msgs)
				signs[signs == 0] = 1.0
				abs_msgs = msgs.abs()

				# compute overall sign product per batch
				sign_prod = torch.prod(signs, dim=0)  # (batch,)

				# compute min1 and min2 along variables
				min_vals, min_idx = torch.min(abs_msgs, dim=0)  # (batch,)

				# To get second minimum per batch, mask the min positions
				# Build masked abs with large value at min positions
				d = abs_msgs.shape[0]
				big = torch.full_like(abs_msgs, float("inf"))
				# For each batch, set the min position to inf
				for b in range(abs_msgs.shape[1]):
					big[:, b] = abs_msgs[:, b].clone()
					big[min_idx[b], b] = float("inf")
				second_min, _ = torch.min(big, dim=0)  # (batch,)

				# get per-check scaling/offset for this iteration (scalars)
				alpha_it_i = alphas[it, i]
				beta_it_i = betas[it, i]

				# For each target variable compute message
				for idx_pos, j in enumerate(vars_idx.tolist()):
					sign_excl = sign_prod * signs[idx_pos, :]
					# select min excluding this variable
					use_min = torch.where(min_idx == idx_pos, second_min, min_vals)
					# apply scaling and offset (offset min-sum): max(alpha*min - beta, 0)
					scaled = alpha_it_i * use_min
					magnitude = torch.clamp(scaled - beta_it_i, min=0.0)
					c2v[i, j, :] = sign_excl * magnitude

			# Variable-node update: v2c = llr + sum_{i'!=i} c2v[i',j]
			# sum c2v over checks
			sum_c2v = torch.sum(c2v, dim=0)  # (n, batch)
			for j in range(self.n):
				checks = var_to_checks[j]
				if checks.numel() == 0:
					continue
				post_j = llr[:, j]  # (batch,)
				for i in checks.tolist():
					# sum of all c2v to var j excluding from check i
					sum_excl = sum_c2v[j, :] - c2v[i, j, :]
					v2c[i, j, :] = post_j + sum_excl

		# final posterior LLR: llr + sum_c c2v
		final_c2v_sum = torch.sum(c2v, dim=0)  # (n, batch)
		final_llr = (llr.transpose(0, 1) + final_c2v_sum).transpose(0, 1)  # (batch, n)

		bits = (final_llr < 0).to(torch.int64)
		if single:
			bits = bits.squeeze(0)
			final_llr = final_llr.squeeze(0)

		if return_llr:
			return bits, final_llr
		return bits

	def syndrome(self, bits: torch.Tensor) -> torch.Tensor:
		"""Compute syndrome H * bits^T mod 2.

		bits: shape (n,) or (batch, n)
		returns: syndrome tensor of shape (m,) or (batch, m)
		"""
		single = False
		if bits.dim() == 1:
			bits = bits.unsqueeze(0)
			single = True
		bits = bits.to(self.H.device).float()
		syn = (torch.matmul(bits, self.H.t()) % 2).to(torch.int64)
		if single:
			syn = syn.squeeze(0)
		return syn

	def trainable_parameters(self):
		"""Return a list of (name, shape) for parameters that require gradients.

		Useful to inspect which model parameters are trainable.
		"""
		return [(name, tuple(p.shape)) for name, p in self.named_parameters() if p.requires_grad]


__all__ = ["NeuralMinSumDecoder"]


if __name__ == "__main__":
	# Simple demo: create a small LDPC, train scaling/offset parameters on noisy
	# all-zero codewords and print/save learned weights.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Example small parity-check matrix (m x n)
	H = torch.tensor([
		[1, 1, 1, 0, 0, 0],
		[0, 0, 1, 1, 1, 0],
		[1, 0, 0, 0, 1, 1],
	], dtype=torch.float32)

	model = NeuralMinSumDecoder(H, num_iterations=5, learnable_scaling=True, learnable_beta=True)
	model.to(device)

	# optimizer only trains parameters that require grad (raw_alpha, raw_beta)
	opt = torch.optim.Adam(model.parameters(), lr=1e-2)
	criterion = torch.nn.BCEWithLogitsLoss()

	batch_size = 64
	epochs = 200
	sigma = 0.7  # AWGN noise std

	n = H.shape[1]
	for epoch in range(epochs):
		# all-zero codeword
		x = torch.zeros((batch_size, n), dtype=torch.int64, device=device)
		# BPSK: 0 -> +1
		tx = 1.0 - 2.0 * x.float()
		noise = sigma * torch.randn_like(tx)
		rx = tx + noise
		# LLR for AWGN: 2*y/sigma^2
		llr = (2.0 * rx / (sigma ** 2)).to(device)

		model.train()
		opt.zero_grad()
		_, final_llr = model(llr, return_llr=True)
		# target bits are zeros; probability of 1 is sigmoid(-LLR)
		targets = x.float()
		loss = criterion(-final_llr, targets)
		loss.backward()
		opt.step()

		if (epoch + 1) % 50 == 0 or epoch == 0:
			# sigmoid(raw_alpha) and sigmoid(raw_beta)
			a = model.alpha().detach().cpu().numpy()
			b = model.beta().detach().cpu().numpy()
			print(f"epoch {epoch+1}/{epochs} loss={loss.item():.4f} alpha={a} beta={b}")

	# Final weights
	final_state = model.state_dict()
	# Print final sigmoid(raw_alpha) and sigmoid(raw_beta)
	final_sigmoid_alpha = model.alpha().detach().cpu().numpy()
	final_sigmoid_beta = model.beta().detach().cpu().numpy()
	print("Final sigmoid(raw_alpha):", final_sigmoid_alpha)
	print("Final sigmoid(raw_beta):", final_sigmoid_beta)
	# Save trained weights
	torch.save(final_state, "trained_ldpc_decoder.pth")

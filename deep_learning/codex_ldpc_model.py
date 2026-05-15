"""
Neural LDPC min-sum decoder (PyTorch).

This module implements a trainable min-sum style message-passing decoder
with learnable per-check, per-iteration scaling (alpha) and optional
offset (beta) parameters. The decoder uses a layered schedule (process
checks/rows sequentially and update variable messages immediately), and
stores the parity-check matrix and sparse neighbor lists as buffers to
avoid re-computation.

Design notes:
- Uses the min-sum approximation of belief propagation: each check-to-
	variable message is computed from the minimum (or second minimum)
	of incoming absolute variable-to-check messages combined with the
	sign product of those messages.
- Learnable parameters raw_alpha/raw_beta are kept in the logit domain
	and mapped through a sigmoid to enforce sensible bounds during
	optimization:
		alpha = sigmoid(raw_alpha) in (0, 1]
		beta  = sigmoid(raw_beta)  in [0, 1)
	This keeps the constrained parameters numerically stable.
- The implementation precomputes `check_to_vars` and `var_to_checks`
	neighbor lists (Python lists of 1D LongTensors) to efficiently
	index messages for each edge.

Utilities are provided to toggle alpha/beta between trainable
parameters and non-trainable buffers, to initialize their sigmoid
targets, compute the syndrome, and list trainable parameter shapes.
"""
from typing import Optional

import torch
import torch.nn as nn


class NeuralLayeredFloatpointMinSumDecoder(nn.Module):
	"""Neural Min-Sum LDPC decoder using a layered schedule.

	Args:
		H: parity-check matrix (torch.Tensor or numpy.ndarray) with shape (m, n).
		   Non-zero entries indicate edges between check nodes (rows) and
		   variable nodes (columns).
		num_iterations: number of layered message-passing iterations (passes over all checks).
		learnable_scaling: if True, raw_alpha is an nn.Parameter and will be trained;
			otherwise raw_alpha is stored as a buffer (non-trainable).
		learnable_beta: if True, raw_beta is an nn.Parameter and will be trained;
			otherwise raw_beta is stored as a buffer (non-trainable).

	Notes:
	- The actual scaling/offset used in message updates are obtained by
	  applying a sigmoid to the stored raw parameters. Both alpha and beta
	  have shape (num_iterations, m) -- i.e. one value per iteration and per
	  check node.
	- Layered scheduling updates check nodes sequentially and immediately
	  updates the corresponding variable-to-check messages, which often
	  improves convergence speed compared to flooding schedules.
	"""

	def __init__(self, H, num_iterations: int = 5, learnable_scaling: bool = True, learnable_beta: bool = False, init_alpha=0.7,init_beta=0.0):
		super().__init__()
		H = torch.as_tensor(H)
		assert H.dim() == 2, "H must be a 2D parity-check matrix"
		self.register_buffer("H", H.float())
		self.m, self.n = H.shape
		self.num_iterations = int(num_iterations)
		self.init_alpha = init_alpha
		self.init_beta = init_beta

		# Precompute neighbor indices
		# check_to_vars: list of variable indices (1D LongTensor) for each check
		# var_to_checks: list of check indices (1D LongTensor) for each variable
		# These lists are used to index into dense message tensors efficiently
		# without constructing sparse tensors.
		self.check_to_vars = [torch.nonzero(self.H[i], as_tuple=False).view(-1) for i in range(self.m)]
		self.var_to_checks = [torch.nonzero(self.H[:, j], as_tuple=False).view(-1) for j in range(self.n)]

		# Initialize alpha so that sigmoid(raw_alpha) = 0.7 by default.
		# We maintain a separate alpha for each iteration and each check node
		# with shape (num_iterations, m). raw_alpha stores the logit values
		# (inverse-sigmoid) so we can optimize in the unconstrained domain.
		alpha_init_val = init_alpha
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
		# raw_beta is stored in the logit domain and later mapped with sigmoid
		# to produce beta values in [0, 1). We initialize near 0 by default.
		beta_init_val = init_beta
		# clamp the initial target into (0,1) to avoid infinite logits
		b_v = torch.tensor(beta_init_val).clamp(0.0 + 1e-6, 1.0 - 1e-6)
		beta_inv = torch.log(b_v / (1.0 - b_v))
		raw_beta_init = beta_inv * torch.ones((self.num_iterations, self.m))
		if learnable_beta:
			self.raw_beta = nn.Parameter(raw_beta_init)
		else:
			self.register_buffer("raw_beta", raw_beta_init)

	def alpha(self):
		# Return scaling factors constrained to (0, 1].
		# raw_alpha is unconstrained; apply sigmoid to map into (0,1) and then
		# clamp a tiny epsilon above zero to ensure strict positivity. The
		# maximum is clamped to 1.0.
		eps = 1e-6
		return torch.sigmoid(self.raw_alpha).clamp(min=eps, max=1.0)

	def beta(self):
		# Return offset factors constrained to [0, 1).
		# raw_beta is mapped with sigmoid into (0,1), but we clamp the
		# maximum slightly below 1.0 to ensure beta < 1.0 numerically.
		eps = 1e-6
		return torch.sigmoid(self.raw_beta).clamp(min=0.0, max=1.0 - eps)

	def set_alpha_trainable(self, trainable: bool = True):
		"""Toggle whether raw_alpha is a trainable parameter.

		When enabling, an existing buffer named 'raw_alpha' is converted into
	
a nn.Parameter so it will appear in model.parameters() and receive
		gradients. When disabling, the Parameter is removed and its value is
		kept as a detached buffer (non-trainable).
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
		"""Toggle whether raw_beta is a trainable parameter.

		Same semantics as set_alpha_trainable: move between buffers and
		Parameters so training can be enabled or disabled at runtime.
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
		"""Initialize raw_alpha so that sigmoid(raw_alpha) == values.

	tThe `values` argument may be:
    - a scalar: the same target for all iterations and checks
    - an iterable of length num_iterations: one target per iteration (broadcast across checks)
    - an iterable of length num_iterations*m: explicit target per (iteration, check)
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
		# inverse sigmoid (logit): raw = log(v/(1-v)). Clamp v into (0,1) first
		# to avoid numerical issues (logit of 0 or 1 is infinite).
		v = v.clamp(1e-6, 1.0 - 1e-6)
		inv = torch.log(v / (1.0 - v))
		if "raw_alpha" in self._parameters:
			with torch.no_grad():
				self.raw_alpha.copy_(inv)
		else:
			with torch.no_grad():
				self._buffers["raw_alpha"].copy_(inv)

	def set_beta_init(self, values):
		"""Initialize raw_beta so that sigmoid(raw_beta) == values.

		Same accepted shapes/semantics as set_alpha_init.
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
		# inverse sigmoid (logit): raw = log(v/(1-v)). Clamp v into (0,1) first
		# to avoid numerical issues (logit of 0 or 1 is infinite).
		v = v.clamp(1e-6, 1.0 - 1e-6)
		inv = torch.log(v / (1.0 - v))
		if "raw_beta" in self._parameters:
			with torch.no_grad():
				self.raw_beta.copy_(inv)
		else:
			with torch.no_grad():
				self._buffers["raw_beta"].copy_(inv)

	def forward(self, llr: torch.Tensor, return_llr: bool = False):
		"""Run the neural layered min-sum decoding algorithm.

		Args:
			llr: input log-likelihood ratios, shape (batch, n) or (n,) for a
				single example. Positive LLR favors bit 0, negative favors bit 1.
			return_llr: if True, also return the final posterior LLR tensor.

		Returns:
			decoded_bits: hard decisions (0/1) with shape (batch, n) or (n,)
			optionally final_llr: posterior LLRs with same shape as decoded_bits

		Implementation details:
		- v2c (variable-to-check) messages are stored in a dense tensor of shape
		  (m, n, batch) where entries corresponding to absent edges remain zero.
		- For each check node, we compute the sign product and the minimum and
		  second-min of absolute incoming v2c messages across connected variables.
		  The outgoing c2v message for an edge is formed from the appropriate
		  min (excluding the target variable) scaled by alpha and offset by beta
		  using clipped subtraction (max(0, alpha*min - beta)).
		- Layered scheduling means c2v messages are computed for one check at a
		  time and v2c messages for the affected variables are immediately
		  updated so subsequent checks in the same iteration see the latest
		  messages.
		"""
		single = False
		if llr.dim() == 1:
			llr = llr.unsqueeze(0)
			single = True

		batch_size = llr.shape[0]
		device = llr.device
		dtype = llr.dtype

		# Move precomputed neighbor index tensors to the current device. Each
		# entry is a 1D LongTensor listing indices of neighboring nodes.
		check_to_vars = [idx.to(device) for idx in self.check_to_vars]
		var_to_checks = [idx.to(device) for idx in self.var_to_checks]

		# Initialize messages: v2c (m, n, batch) stores variable-to-check
		# messages for each (check, var, batch) tuple. Entries for non-edges
		# will remain zero because we index only valid edges below.
		v2c = torch.zeros((self.m, self.n, batch_size), device=device, dtype=dtype)
		mask = (self.H > 0).to(device=device)

		# Initialize v2c with the channel LLRs: for any edge (i,j), set the
		# initial message from variable j to check i to the channel LLR for j.
		for j in range(self.n):
			checks = var_to_checks[j]
			if checks.numel() == 0:
				continue
			# llr[:, j] has shape (batch,), we want shape (d, batch) when
			# broadcasting over checks; PyTorch broadcasting from (batch,) is
			# sufficient when assigning into (d, batch).
			v2c[checks, j, :] = llr[:, j].transpose(0, 0)

		alphas = self.alpha().to(device)
		betas = self.beta().to(device)

		for it in range(self.num_iterations):
			# Layered schedule: process checks (rows) sequentially and update
			# variable-to-check messages for the variables connected to that
			# check immediately. This means later checks in the same iteration
			# will see the updated messages.
			c2v = torch.zeros_like(v2c)

			for i in range(self.m):
				vars_idx = check_to_vars[i]
				if vars_idx.numel() == 0:
					continue
				# messages from variables to this check: shape (d, batch) where d
				# is the degree (#variables connected to this check).
				msgs = v2c[i, vars_idx, :]
				signs = torch.sign(msgs)
				signs[signs == 0] = 1.0
				abs_msgs = msgs.abs()

				# compute overall sign product per batch across incoming messages
				sign_prod = torch.prod(signs, dim=0)  # (batch,)

				# compute min and second-min along variables (per batch). We need
				# the second minimum because the outgoing message for the variable
				# that achieved the minimum must exclude itself.
				min_vals, min_idx = torch.min(abs_msgs, dim=0)  # (batch,)
				# build masked abs for second minimum by temporarily setting the
				# minimal entry to +inf then taking the min again.
				big = torch.full_like(abs_msgs, float("inf"))
				for b in range(abs_msgs.shape[1]):
					big[:, b] = abs_msgs[:, b].clone()
					big[min_idx[b], b] = float("inf")
				second_min, _ = torch.min(big, dim=0)  # (batch,)

				# per-check scaling (alpha) and offset (beta) for this iteration
				alpha_it_i = alphas[it, i]
				beta_it_i = betas[it, i]

				# compute c2v (check-to-variable) messages for all edges of this
				# check. For each target variable we select either the min or the
				# second-min depending on whether the target variable achieved the
				# minimum.
				use_min = torch.where(min_idx.unsqueeze(0) == torch.arange(len(vars_idx), device=min_idx.device).unsqueeze(1),
					second_min.unsqueeze(0), min_vals.unsqueeze(0))  # (d, batch)
				scaled = alpha_it_i * use_min
				magnitude = torch.clamp(scaled - beta_it_i, min=0.0)
				sign_excl = sign_prod.unsqueeze(0) * signs
				c2v_i = sign_excl * magnitude  # (d, batch)
				c2v[i, vars_idx, :] = c2v_i

				# update variable-to-check messages for these variables using the
				# freshly computed c2v values. For each variable we compute the
				# sum of all incoming c2v messages and then add the channel LLR to
				# produce the v2c messages sent to each connected check.
				sum_all = torch.sum(c2v[:, vars_idx, :], dim=0)  # (d, batch)
				# llr for these variables: (batch, d) -> transpose to (d, batch)
				llr_sub = llr[:, vars_idx].transpose(0, 1)
				# v2c for check i: llr + sum_{other checks} c2v = llr + (sum_all - c2v_i)
				v2c_i = llr_sub + (sum_all - c2v_i)
				v2c[i, vars_idx, :] = v2c_i

		# final posterior LLR: channel llr + sum over checks of c2v messages
		final_c2v_sum = torch.sum(c2v, dim=0)  # (n, batch)
		final_llr = (llr.transpose(0, 1) + final_c2v_sum).transpose(0, 1)  # (batch, n)

		bits = (final_llr < 0).to(torch.int64)
		if single:
			bits = bits.squeeze(0)
			final_llr = final_llr.squeeze(0)

		if return_llr:
			return bits, final_llr
		return bits,final_llr

	def syndrome(self, bits: torch.Tensor) -> torch.Tensor:
		"""Compute the binary syndrome: H * bits^T (mod 2).

		bits may be a single vector of shape (n,) or a batch of vectors with
		shape (batch, n). Bits are interpreted as integers (0/1). The returned
		syndrome has shape (m,) or (batch, m) correspondingly.
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

		This is a convenience for debugging/inspection to see which parameters
		are currently trainable (e.g., raw_alpha/raw_beta when enabled).
		"""
		return [(name, tuple(p.shape)) for name, p in self.named_parameters() if p.requires_grad]


__all__ = ["NeuralLayeredFloatpointMinSumDecoder"]


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

	model = NeuralLayeredFloatpointMinSumDecoder(H, num_iterations=5, learnable_scaling=True, learnable_beta=True)
	model.to(device)

	# optimizer only trains parameters that require grad (raw_alpha, raw_beta)
	opt = torch.optim.Adam(model.parameters(), lr=1e-2)
	criterion = torch.nn.BCEWithLogitsLoss()

	batch_size = 64
	epochs = 200
	sigma = 0.7  # AWGN noise std

	# track best parameters (lowest loss)
	best_loss = float("inf")
	best_state = None

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

		# save best parameters (sigmoid alpha/beta) by lowest loss
		if loss.item() < best_loss:
			best_loss = loss.item()
			# store CPU detached state dict
			best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

		if (epoch + 1) % 50 == 0 or epoch == 0:
			# sigmoid(raw_alpha) and sigmoid(raw_beta)
			a = model.alpha().detach().cpu().numpy()
			b = model.beta().detach().cpu().numpy()
			print(f"epoch {epoch+1}/{epochs} loss={loss.item():.4f} alpha={a} beta={b}")

	# Load and use the best parameters seen during training
	if best_state is not None:
		model.load_state_dict(best_state)
	# Final weights (best)
	final_state = model.state_dict()
	# Print final sigmoid(raw_alpha) and sigmoid(raw_beta)
	final_sigmoid_alpha = model.alpha().detach().cpu().numpy()
	final_sigmoid_beta = model.beta().detach().cpu().numpy()
	print("Best epoch loss:", best_loss)
	print("Final sigmoid(raw_alpha):", final_sigmoid_alpha)
	print("Final sigmoid(raw_beta):", final_sigmoid_beta)
	# Save trained weights (best)
	torch.save(final_state, "trained_ldpc_decoder.pth")

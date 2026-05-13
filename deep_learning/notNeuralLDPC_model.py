import torch
import torch.nn as nn

class NeuralnrLDPC(nn.Module):
    def __init__(self, H, Zc, iterations=32):
        super().__init__()

        self.H = H
        self.Zc = Zc
        self.iterations = iterations

        if isinstance(H, torch.Tensor):
            self.register_buffer("H_tensor", H.float())
            self.check_nodes, self.variable_nodes = self.H_tensor.shape
            self.device = self.H_tensor.device
        else:
            self.H_tensor = torch.tensor(H, dtype=torch.float32)
            self.check_nodes, self.variable_nodes = self.H_tensor.shape
            self.device = torch.device("cpu")

        self.H_mask = (self.H_tensor != 0).float()

        self.check_neighbors = [torch.where(self.H_mask[i] > 0)[0].to(self.device)
                                for i in range(self.check_nodes)]
        self.var_neighbors = [torch.where(self.H_mask[:, j] > 0)[0].to(self.device)
                              for j in range(self.variable_nodes)]

        self.alpha = nn.Parameter(torch.ones(iterations, dtype=torch.float32))
        self.beta = nn.Parameter(torch.ones(iterations, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.ones(iterations, dtype=torch.float32))

    def forward(self, llr):
        if llr.dim() == 1:
            llr = llr.unsqueeze(0)

        llr = llr.to(self.device).float()
        batch_size = llr.shape[0]

        v2c = self._initialize_messages(batch_size, llr)
        c2v = torch.zeros_like(v2c)

        for t in range(self.iterations):
            c2v = self._check_node_update(v2c, t)
            v2c = self._variable_node_update(c2v, llr, t)

        final_llr = self.gamma[-1] * llr + c2v.sum(dim=1)
        hard_decision = self._hard_decision(final_llr)
        return final_llr, hard_decision

    def _initialize_messages(self, batch_size, llr):
        v2c = torch.zeros(batch_size, self.check_nodes, self.variable_nodes,
                          device=self.device, dtype=torch.float32)
        for j in range(self.variable_nodes):
            checks = self.var_neighbors[j]
            if checks.numel() == 0:
                continue
            v2c[:, checks, j] = llr[:, j].unsqueeze(1)
        return v2c

    def _check_node_update(self, v2c, t):
        c2v = torch.zeros_like(v2c)
        for i in range(self.check_nodes):
            neigh = self.check_neighbors[i]
            if neigh.numel() == 0:
                continue

            messages = v2c[:, i, neigh]
            sign = torch.sign(messages)
            sign[sign == 0] = 1.0
            abs_messages = torch.abs(messages)

            if neigh.numel() == 1:
                c2v[:, i, neigh[0]] = 0.0
                continue

            total_sign = torch.prod(sign, dim=1, keepdim=True)
            for idx, j in enumerate(neigh):
                excluded = abs_messages.clone()
                excluded[:, idx] = float("inf")
                min_except = torch.min(excluded, dim=1).values
                c2v[:, i, j] = self.alpha[t] * total_sign[:, 0] * sign[:, idx] * min_except

        return c2v

    def _variable_node_update(self, c2v, llr, t):
        v2c = torch.zeros_like(c2v)
        for j in range(self.variable_nodes):
            neigh = self.var_neighbors[j]
            if neigh.numel() == 0:
                continue

            incoming = c2v[:, neigh, j]
            summed = incoming.sum(dim=1, keepdim=True)
            v2c[:, neigh, j] = self.beta[t] * llr[:, j].unsqueeze(1) + (summed - incoming)

        return v2c

    def _hard_decision(self, llr):
        return (llr < 0).long()

    def syndrome(self, bits):
        if bits.dim() == 1:
            bits = bits.unsqueeze(0)

        bits = bits.to(self.device).float()
        parity = torch.remainder(torch.matmul(bits, self.H_tensor.t()), 2)
        return parity.sum(dim=1) == 0

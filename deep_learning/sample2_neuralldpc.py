import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Define the NeuralLDPC Model (as discussed previously) ---
class NeuralLDPC(nn.Module):
    def __init__(self, H, iterations=5):
        super().__init__()
        self.H = torch.tensor(H).float()
        self.num_checks, self.num_vars = self.H.shape
        self.iterations = iterations
        # Trainable weights for each iteration
        self.weights = nn.Parameter(torch.full((iterations, self.num_checks, self.num_vars), 0.8))
        self.mask = (self.H > 0).float()

    def forward(self, received_llr):
        # received_llr shape: [batch_size, num_vars]
        v2c = received_llr.unsqueeze(1).repeat(1, self.num_checks, 1) * self.mask
        
        for i in range(self.iterations):
            # Check Node Update
            signs = torch.sign(v2c + 1e-9)
            abs_v2c = torch.abs(v2c) + (1e9 * (1 - self.mask))
            min_val, _ = torch.min(abs_v2c, dim=2, keepdim=True)
            
            c2v = torch.prod(signs, dim=2, keepdim=True) * signs * min_val
            c2v = c2v * self.weights[i] * self.mask

            # Variable Node Update
            total_sum = received_llr.unsqueeze(1) + torch.sum(c2v, dim=1, keepdim=True)
            v2c = (total_sum - c2v) * self.mask

        return received_llr + torch.sum(c2v, dim=1)

# --- 2. Setup: Simulation and Training ---

# Example Toy Parity Check Matrix (H) - 7x4 Hamming code-style
# Replace this with your actual 5G BG1/BG2 matrix
H_matrix = np.array([
    [1, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1]
])

model = NeuralLDPC(H_matrix, iterations=5)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss() # Good for LLR inputs

# --- 3. Training Loop ---
print("Starting Training...")
for epoch in range(500):
    # Assume all-zero codeword for training (standard practice for LDPC)
    # Target: 0 (which represents binary '0')
    batch_size = 32
    num_vars = H_matrix.shape[1]
    targets = torch.zeros((batch_size, num_vars)) 
    
    # Simulate AWGN Channel: Add noise to a signal of +1 (represents '0')
    snr_db = 3.0
    sigma = 10**(-snr_db/20)
    noise = torch.randn(batch_size, num_vars) * sigma
    received_signal = 1.0 + noise # BPSK: 0 -> +1, 1 -> -1
    
    # Convert to LLR: L = 2*y / sigma^2
    input_llr = (2 * received_signal) / (sigma**2)
    
    # Forward Pass
    optimizer.zero_grad()
    outputs = model(input_llr)
    
    # Loss calculation (comparing decoded LLRs to the transmitted zeros)
    loss = criterion(outputs, targets)
    
    # Backward Pass
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- 4. Inference / Testing ---
model.eval()
with torch.no_grad():
    test_noise = torch.randn(1, num_vars) * 0.8
    test_input = (2 * (1.0 + test_noise)) / (0.8**2)
    decoded_llr = model(test_input)
    hard_decision = (decoded_llr < 0).int() # LLR < 0 means bit '1'
    print(f"\nDecoded Bits: {hard_decision}")

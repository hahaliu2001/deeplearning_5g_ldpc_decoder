import torch
import torch.nn as nn

class HardwareFriendlyNeuralLDPC(nn.Module):
    def __init__(self, bg_matrix, z_factor, iterations=5):
        super().__init__()
        # bg_matrix: The 5G Base Graph (BG1 or BG2) containing -1 or shift values
        # z_factor: The lifting size (e.g., 4, 8, ... 384)
        self.iterations = iterations
        self.z = z_factor
        self.bg = torch.tensor(bg_matrix)
        self.num_bg_rows, self.num_bg_cols = self.bg.shape
        
        # MASK: 1 where Base Graph entry >= 0 (valid sub-matrix)
        self.mask_bg = (self.bg >= 0).float()
        
        # WEIGHTS: One weight per valid sub-matrix per iteration
        # Instead of (Checks x Vars), we only store (BG_Rows x BG_Cols)
        self.submatrix_weights = nn.Parameter(
            torch.full((iterations, self.num_bg_rows, self.num_bg_cols), 0.8)
        )

    def forward(self, received_llr):
        batch_size = received_llr.size(0)
        # Reshape LLRs to match sub-matrix structure [Batch, BG_Cols, Z]
        llr_reshaped = received_llr.view(batch_size, self.num_bg_cols, self.z)
        
        # Initialize v2c [Batch, BG_Rows, BG_Cols, Z]
        v2c = llr_reshaped.unsqueeze(1).repeat(1, self.num_bg_rows, 1, 1)
        # Apply Base Graph mask
        v2c = v2c * self.mask_bg.unsqueeze(0).unsqueeze(-1)

        for i in range(self.iterations):
            # --- Check Node Update ---
            signs = torch.sign(v2c + 1e-9)
            abs_v2c = torch.abs(v2c) + (1e9 * (1 - self.mask_bg.unsqueeze(0).unsqueeze(-1)))
            
            # Min over the BG_Cols dimension
            min_val, _ = torch.min(abs_v2c, dim=2, keepdim=True)
            
            # Neural Scaling: Multiply by the shared weight for this iteration's sub-matrix
            # weight shape: [1, 1, BG_Cols, 1] after indexing
            current_weights = self.submatrix_weights[i].unsqueeze(0).unsqueeze(-1)
            c2v = torch.prod(signs, dim=2, keepdim=True) * signs * min_val * current_weights
            c2v = c2v * self.mask_bg.unsqueeze(0).unsqueeze(-1)

            # --- Variable Node Update ---
            total_sum = llr_reshaped.unsqueeze(1) + torch.sum(c2v, dim=1, keepdim=True)
            v2c = (total_sum - c2v) * self.mask_bg.unsqueeze(0).unsqueeze(-1)

        # Final soft output
        decoded_llr = llr_reshaped + torch.sum(c2v, dim=1)
        return decoded_llr.view(batch_size, -1)


# 1. Setup Parameters
z_factor = 4
iterations = 5
batch_size = 64
learning_rate = 0.001
epochs = 1000

import torch.nn as nn
import torch.optim as optim
import numpy as np
# Use the bg2_example from the previous turn
# model = HardwareFriendlyNeuralLDPC(bg2_example, z_factor, iterations)
# --- 1. Define a simplified 5G NR Base Graph 2 (Partial Matrix) ---
# -1 indicates no connection. Values >= 0 are the cyclic shift (for lifting).
# This is a small slice of BG2 for demonstration.
bg2_example = np.array([
    [ 0,  1, -1, -1,  0, -1, -1, -1, -1, -1],
    [22, -1,  0, -1, -1,  0, -1, -1, -1, -1],
    [ 6, -1, -1,  0, -1, -1,  0, -1, -1, -1],
    [ 2, -1, 24, -1, -1, -1, -1,  0, -1, -1],
    [-1,  1, -1, -1, -1, -1, -1, -1,  0, -1],
    [-1, -1, 25, -1, -1, -1, -1, -1, -1,  0]
])

# Define lifting factor Z (how many bits per sub-matrix)
z_factor = 4 
num_vars = bg2_example.shape[1] * z_factor

# --- 2. Initialize the model ---
# We'll use 5 iterations, which is a common target for low-latency hardware.
model = HardwareFriendlyNeuralLDPC(bg2_example, z_factor, iterations=5)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# BCEWithLogitsLoss is ideal because it handles raw LLRs (logits)
criterion = nn.BCEWithLogitsLoss() 

print("Starting Hardware-Friendly Training...")



for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # 2. Simulate Channel (Training on All-Zero Codeword)
    # Binary 0 is mapped to +1.0 in BPSK
    num_vars = model.num_bg_cols * z_factor
    targets = torch.zeros(batch_size, num_vars) # Target bits are all 0
    
    # Vary SNR during training to make the model robust (e.g., 2dB to 6dB)
    snr_db = torch.FloatTensor(batch_size, 1).uniform_(2.0, 6.0)
    sigma = 10**(-snr_db / 20)
    
    noise = torch.randn(batch_size, num_vars) * sigma
    received = 1.0 + noise
    
    # 3. Convert to LLR
    input_llr = (2 * received) / (sigma**2)
    
    # 4. Forward Pass
    output_llr = model(input_llr)
    
    # 5. Loss & Backpropagation
    # We compare decoded LLRs against the target (all zeros)
    loss = criterion(output_llr, targets)
    loss.backward()
    
    # Optional: Gradient clipping (helps with Min-Sum stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

print("Training Complete.")

# 6. Save Weights for your Product
# These are the fixed constants you will put into your FPGA/ASIC LUT
torch.save(model.state_dict(), "nms_weights.pth")
print("Weights saved to nms_weights.pth")

# --- 3. Simulate Noisy Data (AWGN Channel) ---
batch_size = 2
# For training, we assume all-zero codeword (BPSK mapped to +1.0)
transmitted_signal = torch.ones(batch_size, num_vars)

# Add noise for roughly 4dB SNR
sigma = 0.6
noise = torch.randn(batch_size, num_vars) * sigma
received_signal = transmitted_signal + noise

# Convert received signal to LLR (Input for the Neural Decoder)
# LLR = 2y / sigma^2
input_llr = (2 * received_signal) / (sigma**2)

# --- 4. Inference ---
model.eval()
with torch.no_grad():
    output_llr = model(input_llr)
    
    # Hard decision: if LLR > 0, bit is 0; if LLR < 0, bit is 1.
    decoded_bits = (output_llr < 0).int()

print(f"Input LLR shape: {input_llr.shape}")   # [Batch, 40]
print(f"Output LLR shape: {output_llr.shape}") # [Batch, 40]
print(f"Decoded bits (first batch):\n{decoded_bits[0]}")

# --- 5. Exporting Weights for Product (FPGA/ASIC) ---
# After training, you would extract these and put them in a Look-Up Table (LUT).
learned_weights = model.submatrix_weights.data
print(f"\nWeight matrix shape (Iter x BG_Rows x BG_Cols): {learned_weights.shape}")

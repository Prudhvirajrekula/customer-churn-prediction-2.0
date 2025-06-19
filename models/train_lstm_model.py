import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Load sequences.json (adapted format)
with open("sequences.json", "r") as f:
    raw_data = json.load(f)


# Extract features
sequence_data = [entry["sequence"] for entry in raw_data]
days_since_last_order = [entry["days_since_last_order"] for entry in raw_data]
churn_labels = [entry["churn"] for entry in raw_data]
ltv_values = [entry["ltv"] for entry in raw_data]

# Normalize LTV for regression stability
ltv_values = np.array(ltv_values)
ltv_mean = ltv_values.mean()
ltv_std = ltv_values.std()
ltv_values = (ltv_values - ltv_mean) / ltv_std


class MultiTaskDataset(Dataset):
    def __init__(self, sequences, days, churn, ltv):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.days = torch.tensor(days, dtype=torch.float32).unsqueeze(1)
        self.churn = torch.tensor(churn, dtype=torch.float32).unsqueeze(1)
        self.ltv = torch.tensor(ltv, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        day = self.days[idx]
        x = torch.cat([seq, day], dim=0)  # Combined input
        return x, self.churn[idx], self.ltv[idx]


class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MultiTaskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc_churn = nn.Linear(hidden_size + 1, 1)
        self.fc_ltv = nn.Linear(hidden_size + 1, 1)

    def forward(self, x):
        seq_len = x.shape[1] - 1
        seq_input = x[:, :seq_len].unsqueeze(-1)
        day_feature = x[:, -1].unsqueeze(1)
        _, (hidden, _) = self.lstm(seq_input)
        hidden = hidden.squeeze(0)
        combined = torch.cat([hidden, day_feature], dim=1)
        churn_output = torch.sigmoid(self.fc_churn(combined))
        ltv_output = self.fc_ltv(combined)
        return churn_output, ltv_output


# Prepare dataset
dataset = MultiTaskDataset(sequence_data, days_since_last_order, churn_labels, ltv_values)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = MultiTaskLSTM(input_size=1, hidden_size=32)
criterion_churn = nn.BCELoss()
criterion_ltv = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for x, y_churn, y_ltv in dataloader:
        optimizer.zero_grad()
        pred_churn, pred_ltv = model(x)
        loss_churn = criterion_churn(pred_churn, y_churn)
        loss_ltv = criterion_ltv(pred_ltv, y_ltv)
        loss = loss_churn + loss_ltv
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "models/lstm_multitask_model.pt")
print("✅ Model saved to models/lstm_multitask_model.pt")

# Also save normalization parameters
np.save("models/ltv_normalization.npy", np.array([ltv_mean, ltv_std]))
print("✅ Normalization values saved to models/ltv_normalization.npy")

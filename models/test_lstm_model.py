import torch
import torch.nn as nn
import numpy as np
import json

# Define the same model architecture
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


# Load model
model = MultiTaskLSTM(input_size=1, hidden_size=32)
model.load_state_dict(torch.load("models/lstm_multitask_model.pt"))
model.eval()

# Load normalization values
ltv_mean, ltv_std = np.load("models/ltv_normalization.npy")

# Load sequence data
with open("sequences.json", "r") as f:
    raw_data = json.load(f)

# Inference on a few samples
print("----- PREDICTIONS -----")
for entry in raw_data[:5]:  # First 5 samples
    seq = entry["sequence"]
    day = entry["days_since_last_order"]
    true_churn = entry["churn"]
    true_ltv = entry["ltv"]

    # Prepare input
    x = torch.tensor(seq + [day], dtype=torch.float32).unsqueeze(0)

    # Predict
    with torch.no_grad():
        pred_churn, pred_ltv = model(x)
        churn_pred_label = int(pred_churn.item() > 0.5)
        ltv_pred_value = pred_ltv.item() * ltv_std + ltv_mean

    print(f"Churn: actual={true_churn} | predicted={churn_pred_label} | Prob={pred_churn.item():.3f}")
    print(f"LTV: actual={true_ltv:.2f} | predicted={ltv_pred_value:.2f}")
    print("---")

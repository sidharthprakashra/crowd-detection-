import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from joblib import dump
import os

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv("crowd_dataset.csv")
# Normalize labels
df["day_type"] = df["day_type"].str.lower().str.strip()
df["location"] = df["location"].str.lower().str.strip()
df["time_slot"] = df["time_slot"].str.lower().str.strip()

# ======================================================
# LABEL ENCODING
# ======================================================

day_encoder = LabelEncoder()
location_encoder = LabelEncoder()
time_encoder = LabelEncoder()

df["day_type"] = day_encoder.fit_transform(df["day_type"])
df["location"] = location_encoder.fit_transform(df["location"])
df["time_slot"] = time_encoder.fit_transform(df["time_slot"])

# Save encoders
os.makedirs("artifacts", exist_ok=True)

dump(day_encoder, "artifacts/day_encoder.pkl")
dump(location_encoder, "artifacts/location_encoder.pkl")
dump(time_encoder, "artifacts/time_encoder.pkl")

# ======================================================
# INPUTS
# ======================================================

X = df[[
    "day_type",
    "location",
    "time_slot"
]].values

y = df["crowd_count"].values

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# TENSORS
# ======================================================

X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ======================================================
# MODEL
# ======================================================

class CrowdPredictor(nn.Module):

    def __init__(self):

        super().__init__()

        self.day_emb = nn.Embedding(10, 4)
        self.location_emb = nn.Embedding(20, 8)
        self.time_emb = nn.Embedding(10, 4)

        self.fc = nn.Sequential(

            nn.Linear(16, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        day = self.day_emb(x[:, 0])
        location = self.location_emb(x[:, 1])
        time = self.time_emb(x[:, 2])

        x = torch.cat(
            [day, location, time],
            dim=1
        )

        return self.fc(x)

# ======================================================
# TRAINING
# ======================================================

model = CrowdPredictor()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 300

for epoch in range(epochs):

    model.train()

    outputs = model(X_train).squeeze()

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:

        mae = mean_absolute_error(
            y_train.detach().numpy(),
            outputs.detach().numpy()
        )

        print(
            f"Epoch {epoch} | "
            f"Loss: {loss.item():.2f} | "
            f"MAE: {mae:.2f}"
        )

# ======================================================
# SAVE MODEL
# ======================================================

torch.save(
    model.state_dict(),
    "artifacts/best_model.pt"
)

print("\nModel saved!")

# ======================================================
# TEST
# ======================================================

model.eval()

with torch.no_grad():

    preds = model(X_test).squeeze()

    mae = mean_absolute_error(
        y_test.numpy(),
        preds.numpy()
    )

print(f"\nTest MAE: {mae:.2f}")
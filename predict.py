import torch
import torch.nn as nn
from joblib import load


day_encoder = load("artifacts/day_encoder.pkl")
location_encoder = load("artifacts/location_encoder.pkl")
time_encoder = load("artifacts/time_encoder.pkl")


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


model = CrowdPredictor()

model.load_state_dict(
    torch.load(
        "artifacts/best_model.pt",
        map_location="cpu"
    )
)

model.eval()


def predict_crowd(
    day_type,
    location,
    time_slot
):

    day_type = day_type.lower().strip()
    location = location.lower().strip()
    time_slot = time_slot.lower().strip()


    day_encoded = day_encoder.transform([day_type])[0]
    location_encoded = location_encoder.transform([location])[0]
    time_encoded = time_encoder.transform([time_slot])[0]

    x = torch.tensor([[
        day_encoded,
        location_encoded,
        time_encoded
    ]], dtype=torch.long)

    with torch.no_grad():

        prediction = model(x).item()

    prediction = max(0, int(prediction))

    if prediction <= 20:
        level = "LOW"

    elif prediction <= 60:
        level = "MEDIUM"

    else:
        level = "HIGH"

    return prediction, level


print("\n🔥 CROWD FORECAST SYSTEM 🔥\n")


print("Select Day Type:")
print("0 -> normal_days")
print("1 -> exam_special_days")

day_choice = int(input("\nEnter choice: "))

day_map = {
    0: "normal_days",
    1: "exam_special_days"
}

day_type = day_map[day_choice]


print("\nSelect Location:")
print("0 -> canteen")
print("1 -> gate1")
print("2 -> gate2")
print("3 -> gate3")
print("4 -> ground")
print("5 -> open_auditorium")

location_choice = int(input("\nEnter choice: "))

location_map = {
    0: "canteen",
    1: "gate1",
    2: "gate2",
    3: "gate3",
    4: "ground",
    5: "open_auditorium"
}

location = location_map[location_choice]


print("\nSelect Time Slot:")
print("0 -> morning")
print("1 -> afternoon")
print("2 -> evening")

time_choice = int(input("\nEnter choice: "))

time_map = {
    0: "morning",
    1: "afternoon",
    2: "evening"
}

time_slot = time_map[time_choice]
try:

    count, level = predict_crowd(
        day_type,
        location,
        time_slot
    )

    print("\n=============================")
    print(f"Estimated Crowd Count : {count}")
    print(f"Crowd Density Level   : {level}")
    print("=============================\n")

except Exception as e:

    print("\nERROR:")
    print(e)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pickle

# Chemin relatif vers le CSV
data_path = "../donnees_boursieres_nettoyees.csv"

df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['Close_TSLA']].dropna()

# Mise à l'échelle
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Création des séquences (fenêtre = 30)
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 30
X, y = create_sequences(data_scaled, window_size)

# Division train/test (80% / 20%)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]

# Conversion en tenseurs
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

# Définition du modèle LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

lstm_model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
num_epochs = 100

# Entraînement
for epoch in range(num_epochs):
    lstm_model.train()
    optimizer.zero_grad()
    output = lstm_model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# Sauvegarde du modèle LSTM
torch.save(lstm_model.state_dict(), "lstm_model.pth")

# Sauvegarde du scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Modèle LSTM sauvegardé dans 'lstm_model.pth' et scaler dans 'scaler.pkl'.")

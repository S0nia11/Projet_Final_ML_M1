# lstm_train_save.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle

# 1. Chargement et prétraitement des données
data_path = "./donnees_boursieres_nettoyees.csv"  # chemin relatif depuis 03_Deploiement_du_modele
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['Close_TSLA']].dropna()

# Mise à l'échelle
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Création des séquences avec une fenêtre de 30 pas
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 30
X, y = create_sequences(data_scaled, window_size)

# 2. Division Train/Validation/Test
n_samples = len(X)
train_end = int(n_samples * 0.7)
val_end = int(n_samples * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
# Nous n'utilisons pas l'ensemble de test ici pour l'entraînement

# Conversion en tenseurs
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32)

# Création d'un DataLoader pour l'entraînement en mini-batchs
batch_size = 64
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 3. Définition du modèle LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialisation des états caché et cellulaire
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

lstm_model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

# 4. Entraînement avec Early Stopping
num_epochs = 100
patience = 10
best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    lstm_model.train()
    train_loss_epoch = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = lstm_model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    train_loss_epoch /= len(train_loader)
    
    # Évaluation sur l'ensemble de validation
    lstm_model.eval()
    with torch.no_grad():
        val_output = lstm_model(X_val_t)
        val_loss = criterion(val_output, y_val_t).item()
    
    scheduler.step(val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_epoch:.6f}, Val Loss: {val_loss:.6f}")
    
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(lstm_model.state_dict(), "lstm_model_best.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping activé")
            break

print("Entraînement terminé. Meilleur modèle sauvegardé dans 'lstm_model_best.pth'.")

# 5. Sauvegarde du scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler sauvegardé dans 'scaler.pkl'.")

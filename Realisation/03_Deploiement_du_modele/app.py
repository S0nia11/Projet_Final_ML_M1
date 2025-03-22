import streamlit as st
import torch
import pickle
import numpy as np
import torch.nn as nn

# Définition de la classe LSTM (identique à celle utilisée lors de l'entraînement)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialisation des états
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Chargement du modèle et du scaler
@st.cache(allow_output_mutation=True)
def load_model():
    model = LSTMModel()
    model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
    model.eval()
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.title("Prédiction du Prix de Clôture de Tesla avec LSTM")
st.write("Entrez les 30 dernières valeurs (par exemple, le cours de clôture) pour prédire le prochain prix.")

# Saisie de 30 valeurs via Streamlit
window_size = 30
input_str = st.text_area("Entrez 30 valeurs séparées par des virgules", 
                           "230.5, 231.1, 229.0, 232.3, 231.8, 230.9, 232.1, 233.5, 234.0, 232.8, 233.2, 234.1, 233.8, 235.0, 234.5, 233.9, 234.3, 235.2, 235.0, 234.8, 235.1, 234.6, 234.9, 235.3, 235.0, 234.7, 235.2, 235.1, 234.8, 235.0")

if st.button("Prédire"):
    try:
        # Conversion de la chaîne en liste de float
        input_data = [float(x.strip()) for x in input_str.split(",")]
        if len(input_data) != window_size:
            st.error(f"Veuillez saisir exactement {window_size} valeurs.")
        else:
            # Mise à l'échelle de la séquence d'entrée
            input_array = np.array(input_data).reshape(-1, 1)
            input_scaled = scaler.transform(input_array)
            input_scaled = input_scaled.reshape(1, window_size, 1)
            
            # Conversion en tenseur et prédiction
            x_input = torch.tensor(input_scaled, dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = model(x_input).item()
            # Inversion de la mise à l'échelle
            pred_original = scaler.inverse_transform([[pred_scaled]])[0][0]
            st.success(f"Prédiction du prochain prix : {pred_original:.2f}")
    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {e}")

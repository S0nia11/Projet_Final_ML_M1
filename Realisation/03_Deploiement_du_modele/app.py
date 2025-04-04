import streamlit as st
import torch
import pickle
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Chargement du modèle et du scaler
# ----------------------------
@st.cache_resource
def load_model():
    # Définition de la classe LSTM (identique à l'entraînement)
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
    model = LSTMModel()
    model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))
    model.eval()
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

st.title("Prédiction du Prix de Clôture de Tesla avec LSTM")
st.write("Saisissez les 30 dernières valeurs (cours de clôture) pour prédire le prochain prix.")

# Zone de saisie pour 30 valeurs
window_size = 30
input_str = st.text_area(
    "Entrez 30 valeurs séparées par des virgules",
    "230.5, 231.1, 229.0, 232.3, 231.8, 230.9, 232.1, 233.5, 234.0, 232.8, 233.2, 234.1, 233.8, 235.0, 234.5, 233.9, 234.3, 235.2, 235.0, 234.8, 235.1, 234.6, 234.9, 235.3, 235.0, 234.7, 235.2, 235.1, 234.8, 235.0"
)

if st.button("Prédire"):
    try:
        # Conversion de la chaîne en liste de float
        input_data = [float(x.strip()) for x in input_str.split(",")]
        if len(input_data) != window_size:
            st.error(f"Veuillez saisir exactement {window_size} valeurs.")
        else:
            # Mise à l'échelle
            input_array = np.array(input_data).reshape(-1, 1)
            input_scaled = scaler.transform(input_array)
            input_scaled = input_scaled.reshape(1, window_size, 1)
            
            # Conversion en tenseur et prédiction
            x_input = torch.tensor(input_scaled, dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = model(x_input).item()
            
            # Inversion de la mise à l'échelle pour obtenir la valeur prédite dans l'échelle d'origine
            pred_original = scaler.inverse_transform([[pred_scaled]])[0][0]
            
            st.success(f"Prédiction du prochain prix de clôture : {pred_original:.2f}")
            
            # Construction d'une séquence étendue : 30 valeurs saisies + 1 valeur prédite
            extended_series = np.array(input_data + [pred_original])
            extended_index = list(range(1, window_size + 2))  # Indices de 1 à 31

            # Graphique 1 : Série temporelle étendue
            fig1, ax1 = plt.subplots()
            ax1.plot(extended_index, extended_series, marker='o', linestyle='-')
            ax1.set_title("Séquence d'entrée étendue avec la prédiction")
            ax1.set_xlabel("Index (pas de temps)")
            ax1.set_ylabel("Prix de clôture")
            ax1.axvline(x=window_size, color='gray', linestyle='--', label='Fin de la fenêtre')
            ax1.legend()
            st.pyplot(fig1)
            
            # Graphique 2 : Histogramme avec densité
            fig2, ax2 = plt.subplots()
            sns.histplot(extended_series, kde=True, ax=ax2)
            ax2.set_title("Histogramme de la séquence étendue")
            ax2.set_xlabel("Prix de clôture")
            st.pyplot(fig2)
            
            # Graphique 3 : Boxplot
            fig3, ax3 = plt.subplots()
            sns.boxplot(x=extended_series, ax=ax3)
            ax3.set_title("Boxplot de la séquence étendue")
            ax3.set_xlabel("Prix de clôture")
            st.pyplot(fig3)
    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {e}")

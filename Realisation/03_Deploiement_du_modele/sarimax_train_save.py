import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

# Charger les données et préparer la série
df = pd.read_csv("../donnees_boursieres_nettoyees.csv", parse_dates=['Date'], index_col='Date')
ts_data = df['Close_TSLA'].dropna()

# Ajustement du modèle SARIMAX (exemple : SARIMAX(1,1,1)x(1,1,1,7))
sarimax_model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarimax_fit = sarimax_model.fit(disp=False)

# Sauvegarde du modèle SARIMAX dans un fichier pickle
with open("sarimax_model.pkl", "wb") as f:
    pickle.dump(sarimax_fit, f)

print("Modèle SARIMAX sauvegardé dans 'sarimax_model.pkl'")

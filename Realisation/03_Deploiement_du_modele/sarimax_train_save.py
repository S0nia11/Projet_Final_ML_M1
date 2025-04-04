# sarimax_train_save.py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pickle

# 1. Chargement des données
data_path = "../donnees_boursieres_nettoyees.csv"  # Chemin relatif depuis 03_Deploiement_du_modele
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
df = df.dropna(subset=['Close_TSLA'])
ts = df['Close_TSLA']

# 2. Test de stationnarité
adf_result = adfuller(ts)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f"Critique {key}: {value:.3f}")

# 3. Transformation si nécessaire
if adf_result[1] > 0.05:
    # La série n'est pas stationnaire, on applique une différenciation
    ts_diff = ts.diff().dropna()
    print("La série n'était pas stationnaire. Différenciation appliquée.")
else:
    ts_diff = ts
    print("La série est stationnaire.")

# 4. Visualisation de la série transformée
plt.figure(figsize=(10,4))
plt.plot(ts_diff)
plt.title("Série (différenciée si nécessaire)")
plt.xlabel("Date")
plt.ylabel("Prix de clôture")
plt.show()

# 5. Ajustement du modèle SARIMAX
# Dans cet exemple, nous utilisons SARIMAX(1,1,1)x(1,1,1,7).
model = SARIMAX(ts_diff, order=(1,1,1), seasonal_order=(1,1,1,7))
sarimax_fit = model.fit(disp=False)
print(sarimax_fit.summary())

# 6. Sauvegarde du modèle dans un fichier pickle
with open("sarimax_model.pkl", "wb") as f:
    pickle.dump(sarimax_fit, f)
print("Modèle SARIMAX sauvegardé dans 'sarimax_model.pkl'.")

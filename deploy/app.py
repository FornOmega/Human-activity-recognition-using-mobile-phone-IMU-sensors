import streamlit as st
import numpy as np
import pickle

# Carica il modello ML
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Configurazione della pagina
st.set_page_config(page_title="ML Prediction HAR App", layout="centered")

st.title("🔮 Machine Learning HAR Prediction App")
st.write("Inserisci i valori per accelerazione e giroscopio per ottenere una previsione sulla tipologia di attività svolta.")

# Etichette per i campi di input
labels = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]


# Creazione della form con 6 campi numerici
input_values = []
for label in labels:
    val = st.number_input(f"{label}", min_value=-1e6, max_value=1e6, step=0.01, value=0.0)
    input_values.append(val)

# Bottone per la predizione
if st.button("Predici"):
    prediction = 'Walking' if model.predict([np.array(input_values)])[0] == 1 else 'Standing'
    st.success(f"📊 Risultato della previsione: **{prediction}**")
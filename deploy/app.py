import streamlit as st
import numpy as np
import pandas as pd
#import pickle
import joblib

# Carica il modello ML
with open("classification_pipeline.joblib", "rb") as file:
    # Load the saved model
    model = joblib.load(file)

# Configurazione della pagina
st.set_page_config(page_title="ML Prediction HAR App", layout="centered")

st.title("🔮 Machine Learning HAR Prediction App")
st.write("Inserisci i valori per accelerazione e giroscopio per ottenere una previsione sulla tipologia di attività svolta.")

# Etichette per i campi di input
labels = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]


# Creazione della form con 6 campi numerici
input_values = []
for label in labels:
    val = st.number_input(label, min_value=-1e6, max_value=1e6, step=0.01, value=0.0)
    input_values.append(val)

print(input_values)

# Bottone per la predizione
if st.button("Predici"):
    # Converto i dati di input in DataFrame
    input_data = pd.DataFrame([input_values], columns=['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ'])
    prediction = 'Walking' if model.predict(input_data)[0] == 1 else 'Standing (Stop)'
    st.success(f"📊 Risultato della previsione: **{prediction}**")
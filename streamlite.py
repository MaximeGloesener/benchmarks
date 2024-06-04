import streamlit as st
import pandas as pd

# Title
st.title('Pytorch Model Benchmarking')
st.write('This app benchmarks all pytorch models on different devices')

# Sidebar for selecting device
st.sidebar.title('Settings')
device = st.sidebar.selectbox('Device', ['GTX3060', 'RTX3090'])

# Load data based on the selected device
file_name = f"{device}.txt"
try:
    data = pd.read_csv(file_name, header=None, names=['Model', 'FPS', 'Nombre de paramètres (M)', 'Taille modèle (MB)', 'Nombre de MACs (M)'])
    st.write(f"Displaying data for: {device}")
    st.write(data)
except FileNotFoundError:
    st.error(f"No data available for {device}")

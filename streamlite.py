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
    data = pd.read_csv(file_name, header=None, names=['Model', 'FPS CPU', 'FPS GPU', 'Nombre de paramètres (M)', 'Taille modèle (MB)', 'Nombre de MACs (M)', 'Max memory used (MB)', 'TOP 1 ACC', 'TOP 5 ACC'])   
    st.write(f"Displaying data for: {device}")
    st.write(data)
except FileNotFoundError:
    st.error(f"No data available for {device}")

# load data for yolo models
file_name = f"benchmark_yolo_{device}.txt"
try:
    data = pd.read_csv(file_name, header=None, names=['Model', 'Mean inference time (ms)', 'FPS', 'Nombre de paramètres (M)', 'Taille modèle (MB)'])
    st.write(f"Displaying data for YOLO models on: {device}")
    st.write(data)
except FileNotFoundError:
    st.error(f"No data available for YOLO models on {device}")

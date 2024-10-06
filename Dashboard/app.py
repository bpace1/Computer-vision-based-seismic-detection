import streamlit as st
import pandas as pd
import numpy as np
import os
from obspy import read
import altair as alt
import pickle
import plotly.graph_objects as go
from typing import Literal
from uuid import uuid4
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Seismic Detection by Space Coders",
    page_icon="🌐🌙",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

st.title('Seismic Detection by Space Coders')

# Mars paths
MARS_TRAINING_PATH: str = 'space_apps_2024_seismic_detection/data/mars/training/data/'
MARS_CATALOGS_PATH: str = 'space_apps_2024_seismic_detection/data/mars/training/catalogs/'
MARS_TEST_PATH: str = 'space_apps_2024_seismic_detection/data/mars/test/data/'
MARS_CATALOG: str = 'Mars_InSight_training_catalog_final.csv'

# Lunar paths
LUNAR_TRAINING_PATH: str = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'
LUNAR_CATALOGS_PATH: str = 'space_apps_2024_seismic_detection/data/lunar/training/catalogs'
LUNAR_TEST_PATH: str = 'space_apps_2024_seismic_detection/data/lunar/test/data/S12_GradeB/'
LUNAR_CATALOG: str = 'apollo12_catalog_GradeA_final.csv'

class Dataset:
    def __init__(self, catalog_pos: pd.Series, type_: Literal['moon', 'mars']) -> None:
        self.dataset_data = catalog_pos
        self.arrival_time = catalog_pos['time_rel(sec)']  # Este es el valor real del inicio del terremoto
        self.filename = self.dataset_data.filename
        self.type_ = type_

    def get_path(self) -> str:
        base_path = MARS_TRAINING_PATH if self.type_ == 'mars' else LUNAR_TRAINING_PATH
        return os.path.join(base_path, self.filename)

    def get_dataframe(self) -> pd.DataFrame:
        ext: str = '.csv' if self.type_ == 'moon' else ''
        return pd.read_csv(self.get_path() + ext)

    def get_arrival_time(self) -> float:
        return self.arrival_time  # Este valor se utilizará como etiqueta durante el entrenamiento
    
    def get_filename(self) -> str:
        return os.path.basename(self.filename)

with open('train_predictions.pickle', 'rb') as f:
    data = pickle.load(f)

def get_datasets(selected_body: str):
    return [data_['dataset'] for data_  in data if data_['type'] == selected_body.lower()]

"""
def plot_seismic_data(df, arrival_time=None, min_=None, max_=None):
    fig = go.Figure()
    if 'rel_time(sec)' in df.columns:
        fig.add_trace(go.Scatter(x=df['rel_time(sec)'], y=df['velocity(c/s)'], mode='lines', name='Velocity'))
        fig.update_layout(title="Seismic Signal", xaxis_title="rel_time(sec)", yaxis_title="velocity(c/s)")
        
    else:
        fig.add_trace(go.Scatter(x=df['time_rel(sec)'], y=df['velocity(m/s)'], mode='lines', name='Velocity'))
        fig.update_layout(title="Seismic Signal", xaxis_title="time_rel(sec)", yaxis_title="velocity(m/s)")
        
    # Add the arrival time line if provided
    if arrival_time is not None:
        fig.add_vline(x=arrival_time, line=dict(color='red'), name='Rel. Arrival')

    if min is not None:
        fig.add_vline(x=min_, line=dict(color='blue'), name='Second Time')
    
    if max is not None:
        fig.add_vline(x=max_, line=dict(color='blue'), name='Second Time')

    return fig

"""

def plot_seismic_data(df, arrival_time=None, min_=None, max_=None):
    fig = go.Figure()
    if 'rel_time(sec)' in df.columns:
        fig.add_trace(go.Scatter(x=df['rel_time(sec)'], y=df['velocity(c/s)'], mode='lines', name='Velocity'))
        fig.update_layout(title="Seismic Signal", xaxis_title="rel_time(sec)", yaxis_title="velocity(c/s)")
        
    else:
        fig.add_trace(go.Scatter(x=df['time_rel(sec)'], y=df['velocity(m/s)'], mode='lines', name='Velocity'))
        fig.update_layout(title="Seismic Signal", xaxis_title="time_rel(sec)", yaxis_title="velocity(m/s)")
    
    # Add the arrival time line if provided
    if arrival_time is not None:
        fig.add_vline(x=arrival_time, line=dict(color='red'), name='Rel. Arrival')

    if min_ is not None:
        fig.add_vline(x=min_, line=dict(color='blue'), name='Second Time')
    
    if max_ is not None:
        fig.add_vline(x=max_, line=dict(color='blue'), name='Second Time')

    if 'time_rel(sec)' in df.columns:
        fig.update_xaxes(range=(df['time_rel(sec)'].min(), df['time_rel(sec)'].max()))
    else:
        fig.update_xaxes(range=(df['rel_time(sec)'].min(), df['rel_time(sec)']))
    
    return fig


with st.sidebar:
    st.title('🌐 Seismic Detection by Space Coders')
    selected_body = st.selectbox(label="Select the heavenly body:", options=['Mars', 'Moon'])
    selected_type_set = st.selectbox(label='Select the type of dataset:', options=['Training', 'Test'])
    datasets = get_datasets(selected_body)
    filenames = [ds.get_filename() for ds in datasets]
    selected_dataset = st.selectbox(label='Select the dataset:', options=filenames)


df_aux, arr_time, min_time, max_time = [(x['dataset'], x['arrival_time'], x['min_'], x['max_']) for x in data if x['dataset'].filename == selected_dataset][0]
df: pd.DataFrame = df_aux.get_dataframe()

st.plotly_chart(plot_seismic_data(df, arrival_time=arr_time, min_=float (min_time), max_=float(max_time)))
st.write(min_time)
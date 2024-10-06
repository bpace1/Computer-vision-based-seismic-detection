import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from src.dataset import Dataset

st.set_page_config(
    page_title="Seismic Detection by Space Coders",
    page_icon="🌐🌙",
    layout="wide",
    initial_sidebar_state="expanded")

c1, c2 = st.columns([1, 5], vertical_alignment="center")

with c1:
    st.image('static/nasa_logo.png', use_column_width=True)
with c2:
    st.title('Seismic Detection by Space Coders')

with open('data/predictions.pickle', 'rb') as f:
    data = pickle.load(f)
    
def get_datasets(selected_body: str):
    return [data_['dataset'] for data_  in data if data_['type'] == selected_body.lower()]

def plot_seismic_data(df: pd.DataFrame, arrival_time=None, min_=None, max_=None):
    fig = go.Figure()
    if 'rel_time(sec)' in df.columns:
        fig.add_trace(go.Line(x=df['rel_time(sec)'].to_numpy(), y=df['velocity(c/s)'], name='Velocity'))
        fig.update_layout(title="Seismic Signal", xaxis_title="rel_time(sec)", yaxis_title="velocity(c/s)")
        
    else:
        fig.add_trace(go.Line(x=df['time_rel(sec)'].to_numpy(), y=df['velocity(m/s)'], name='Velocity'))
        fig.update_layout(title="Seismic Signal", xaxis_title="time_rel(sec)", yaxis_title="velocity(m/s)")
    
    if arrival_time is not None:
        fig.add_vline(x=arrival_time, line=dict(color='red'), name='Rel. Arrival')

    if min_ is not None:
        fig.add_vline(x=min_, line=dict(color='blue'), name='Second Time')
    
    if max_ is not None:
        fig.add_vline(x=max_, line=dict(color='blue'), name='Second Time')

    if 'time_rel(sec)' in df.columns:
        fig.update_xaxes(range=(df['time_rel(sec)'].min(), df['time_rel(sec)'].max()))
    else:
        fig.update_xaxes(range=(df['rel_time(sec)'].min(), df['rel_time(sec)'].max()))
    
    return fig

with st.sidebar:
    st.title('Menu')
    selected_body = st.selectbox(label="Select the heavenly body:", options=['Moon', 'Mars'])
    selected_type_set = st.selectbox(label='Select the type of dataset:', options=['Training', 'Test'], disabled=True)
    datasets = get_datasets(selected_body)
    filenames = [ds.get_filename() for ds in datasets]
    selected_dataset = st.selectbox(label='Select the dataset:', options=filenames)


df_aux, arr_time, min_time, max_time, type_ = [(x['dataset'], x['arrival_time'], x['min_'], x['max_'], x['type']) for x in data if x['dataset'].filename == selected_dataset][0]

df: pd.DataFrame = df_aux.get_dataframe()

col_name: str = 'rel_time(sec)' if type_ == 'mars' else 'time_rel(sec)'

min_time = df[col_name].astype('float64').to_numpy()[min_time]
max_time = df[col_name].astype('float64').to_numpy()[max_time]

st.plotly_chart(plot_seismic_data(df, arrival_time=arr_time, min_=min_time, max_=max_time))

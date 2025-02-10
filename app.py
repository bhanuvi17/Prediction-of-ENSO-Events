import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(
    page_title="ENSO Prediction Dashboard",
    page_icon="🌊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌊 ENSO Prediction Dashboard")
st.markdown("""
This dashboard presents predictions and analysis of the El Niño-Southern Oscillation (ENSO) 
using a trained LSTM model. The Oceanic Niño Index (ONI) is used to identify El Niño and 
La Niña events.
""")

# Function to transform time series data
@st.cache_data
def series_to_supervised(data, n_in=1, n_out=1, n_vars=1, forecast_all=True, dropnan=True):
    cols, names = list(), list()
    if n_vars == 1:
        for i in range(n_in, 0, -1):
            cols.append(data.shift(i))
            names.append(f'var1 (t-{i})')
        cols.append(data)
        names.append('var1 (t)')
        for i in range(1, n_out):
            cols.append(data.shift(-i))
            names.append(f'var1 (t+{i})')
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Load data and model
@st.cache_resource
def load_data_and_model():
    try:
        df_enso = pd.read_csv("ENSO.csv", parse_dates=[0])
        df_enso.set_index('Date', inplace=True)
        model = load_model('model_lstm.keras')
        return df_enso, model
    except Exception as e:
        st.error(f"Error loading data or model: {str(e)}")
        return None, None

# Process data and make predictions
def process_data(df_enso, model):
    # Parameters
    n_in = 12
    n_out = 3
    n_steps = n_in
    n_features = 1

    # Transform data
    df_reframed = series_to_supervised(df_enso['ONI'], n_in, n_out, n_features)

    # Split data
    n = df_reframed.shape[0]
    n_train, n_valid = int(0.8 * n), int(0.1 * n)
    df_test = df_reframed.values[n_train + n_valid:, :]
    x_test, y_test = df_test[:, :-n_out], df_test[:, -n_out:]

    # Scale data
    x_scaler = MinMaxScaler(feature_range=(0,1))
    y_scaler = MinMaxScaler(feature_range=(0,1))

    x_scaler.fit(df_reframed.values[:, :-n_out])
    y_scaler.fit(df_reframed.values[:, -n_out:])

    x_test = x_scaler.transform(x_test)
    y_test = y_scaler.transform(y_test)

    # Reshape input
    x_test = x_test.reshape(x_test.shape[0], n_steps, n_features)

    # Make predictions
    y_hat = model.predict(x_test)
    y_hat = np.round(y_scaler.inverse_transform(y_hat), 1)

    # Prepare data for visualization
    y_start = n_train + n_valid + 1

    y_actual = pd.DataFrame(
        index=df_reframed.index[y_start:],
        data=y_scaler.inverse_transform(y_test)[:-1, 0],
        columns=['Actual']
    )

    y_predict = pd.DataFrame(
        index=df_reframed.index[y_start:],
        data=y_hat[:-1, 0],
        columns=['Predicted']
    )

    y_forecast = pd.DataFrame(
        index=pd.date_range(start=df_reframed.index[-1], periods=n_out, freq='MS'),
        data=y_hat[-1, :],
        columns=['Forecast']
    )

    return y_actual, y_predict, y_forecast

# Create visualizations
def create_prediction_plot(y_actual, y_predict, y_forecast):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=y_actual.index, y=y_actual['Actual'],
                  name='Actual', line=dict(color='black'))
    )

    fig.add_trace(
        go.Scatter(x=y_predict.index, y=y_predict['Predicted'],
                  name='Predicted', line=dict(color='blue'))
    )

    fig.add_trace(
        go.Scatter(x=y_forecast.index, y=y_forecast['Forecast'],
                  name='Forecast', line=dict(color='red'))
    )

    fig.update_layout(
        title='ONI Values: Actual vs Predicted with Forecast',
        xaxis_title='Date',
        yaxis_title='ONI Value',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

def create_error_plot(y_actual, y_predict):
    errors = y_actual['Actual'] - y_predict['Predicted']
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=errors.index, y=errors,
                  mode='lines+markers',
                  name='Prediction Error',
                  line=dict(color='purple'))
    )

    fig.add_trace(
        go.Scatter(x=errors.index, y=[0]*len(errors),
                  line=dict(color='black', dash='dash'),
                  name='Zero Error')
    )

    fig.update_layout(
        title='Prediction Error Over Time',
        xaxis_title='Date',
        yaxis_title='Error (Actual - Predicted)',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

def create_phase_plot(y_actual, y_predict):
    fig = go.Figure()

    # Add horizontal lines for ENSO classification
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.3)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", opacity=0.3)

    fig.add_trace(
        go.Scatter(x=y_actual.index, y=y_actual['Actual'],
                  name='Actual',
                  line=dict(color='black'))
    )

    fig.add_trace(
        go.Scatter(x=y_predict.index, y=y_predict['Predicted'],
                  name='Predicted',
                  line=dict(color='blue'))
    )

    fig.update_layout(
        title='ENSO Phase Classification',
        xaxis_title='Date',
        yaxis_title='ONI Value',
        annotations=[
            dict(x=1.02, y=0.5, xref='paper', yref='y',
                 text='El Niño threshold', showarrow=False),
            dict(x=1.02, y=-0.5, xref='paper', yref='y',
                 text='La Niña threshold', showarrow=False)
        ],
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

# Main app
def main():
    # Load data and model
    df_enso, model = load_data_and_model()
    
    if df_enso is None or model is None:
        return

    # Process data
    y_actual, y_predict, y_forecast = process_data(df_enso, model)

    # Sidebar
    st.sidebar.header("Dashboard Controls")
    plot_type = st.sidebar.selectbox(
        "Select Plot Type",
        ["All Plots", "Prediction Plot", "Error Analysis", "Phase Classification"]
    )

    # Display metrics
    st.sidebar.header("Performance Metrics")
    mse = np.mean((y_actual['Actual'] - y_predict['Predicted'])**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_actual['Actual'] - y_predict['Predicted']))

    st.sidebar.metric("MSE", f"{mse:.4f}")
    st.sidebar.metric("RMSE", f"{rmse:.4f}")
    st.sidebar.metric("MAE", f"{mae:.4f}")

    # Display forecast
    st.sidebar.header("3-Month Forecast")
    for date, value in y_forecast.iterrows():
        st.sidebar.metric(
            date.strftime('%Y-%m'),
            f"{value['Forecast']:.2f}",
            delta=None
        )

    # Main content
    if plot_type == "All Plots" or plot_type == "Prediction Plot":
        st.plotly_chart(create_prediction_plot(y_actual, y_predict, y_forecast), use_container_width=True)
    
    if plot_type == "All Plots" or plot_type == "Error Analysis":
        st.plotly_chart(create_error_plot(y_actual, y_predict), use_container_width=True)
    
    if plot_type == "All Plots" or plot_type == "Phase Classification":
        st.plotly_chart(create_phase_plot(y_actual, y_predict), use_container_width=True)

if __name__ == "__main__":
    main()
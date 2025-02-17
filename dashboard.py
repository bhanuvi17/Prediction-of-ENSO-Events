import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Page Configuration
st.set_page_config(
    page_title="ENSO Event Prediction Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with selective green elements for better visibility
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(120deg, #f0f2f6 0%, #e0e5ec 100%);
    }
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.17);
    }
    /* Red titles and headings ONLY */
    h1, h2, h3 {
        color: #008080 !important;
        font-weight: bold !important;
    }
    /* Green download buttons ONLY */
    .stDownloadButton button {
        background-color: #ff6347 !important;
        color: white !important;
        border: none !important;
    }
    .stDownloadButton button:hover {
        background-color: #1b5e20 !important;
        color: white !important;
    }
    /* Improve date range selector visibility */
    .stDateInput input {
        border: 2px solid #2e7d32 !important;
    }
    </style>
""", unsafe_allow_html=True)

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
    elif forecast_all:
        for i in range(n_in, 0, -1):
            cols.append(data.shift(i))
            names += [f'var{j+1} (t-{i})' for j in range(n_vars)]
        cols.append(data)
        names += [f'var{j+1} (t)' for j in range(n_vars)]
        for i in range(1, n_out):
            cols.append(data.shift(-i))
            names += [f'var{j+1} (t+{i})'for j in range(n_vars)]
    else:
        for i in range(n_in, 0, -1):
            cols.append(data.shift(i))
            names += [f'var{j+1} (t-{i})' for j in range(n_vars)]
        cols.append(data.iloc[:, -1])
        names.append('VAR (t)')
        for i in range(1, n_out):
            cols.append(data.shift(-i).iloc[:,-1])
            names.append(f'VAR (t+{i})')
            
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

@st.cache_resource
def load_data_and_model():
    try:
        df_enso = pd.read_csv("ENSO.csv", parse_dates=[0])
        df_enso.set_index('Date', inplace=True)
        model = load_model('model_lstm.keras')
        return df_enso, model
    except FileNotFoundError:
        st.error("Data or model files not found. Please check file paths.")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None, None

def calculate_metrics(y_actual, y_predict):
    mse = np.mean((y_actual['Actual'] - y_predict['Predicted'])**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_actual['Actual'] - y_predict['Predicted']))
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
    }

def add_download_button(y_actual, y_predict, y_forecast):
    combined_data = pd.concat([y_actual, y_predict, y_forecast], axis=1)
    csv = combined_data.to_csv()
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="enso_predictions.csv",
        mime="text/csv"
    )

def add_date_range_selector(df_enso):
    min_date = df_enso.index.min().to_pydatetime().date()
    max_date = df_enso.index.max().to_pydatetime().date()
    
    # Fix: Convert pandas timestamps to Python date objects for the date picker
    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    if start_date > end_date:
        st.sidebar.error("End date must be after start date")
        end_date = start_date
    
    return (start_date, end_date)

def add_confidence_intervals(fig, y_forecast, std_dev=0.5):
    upper_bound = y_forecast['Forecast'] + 2 * std_dev
    lower_bound = y_forecast['Forecast'] - 2 * std_dev
    
    # Upper Bound - Keep original red color
    fig.add_trace(go.Scatter(
        x=y_forecast.index,
        y=upper_bound,
        mode='lines',
        line=dict(color='red', dash='dot'),  # Keep original red dot line
        name='Upper Bound'
    ))
    
    # Lower Bound - Shaded Confidence Interval
    fig.add_trace(go.Scatter(
        x=y_forecast.index,
        y=lower_bound,
        fill='tonexty',  # This shades the area between lower and upper bounds
        mode='lines',
        line=dict(color='rgba(255,0,0,0.1)'),  # Keep original light red shade
        name='Lower Bound'
    ))  


def add_summary_statistics(df_enso):
    st.sidebar.header("Data Summary")
    stats = {
        "Date Range": f"{df_enso.index.min().strftime('%Y-%m')} to {df_enso.index.max().strftime('%Y-%m')}",
        "Average ONI": f"{df_enso['ONI'].mean():.2f}",
        "Max ONI": f"{df_enso['ONI'].max():.2f}",
        "Min ONI": f"{df_enso['ONI'].min():.2f}"
    }
    for key, value in stats.items():
        st.sidebar.text(f"{key}: {value}")

def add_plot_export(fig, key_suffix=""):
    # Export plot as HTML instead of PNG
    html = fig.to_html()
    st.download_button(
        label="Download Plot as HTML",
        data=html,
        file_name=f"enso_plot_{key_suffix}.html",  # Unique file name
        mime="text/html",
        key=f"plot_export_{key_suffix}"  # Unique key
    )


def process_data(df_enso, model, date_range=None):
    n_in = 12
    n_out = 3
    n_features = 1
    n_steps = n_in
    
    # Filter by date range if provided
    if date_range:
        start_date, end_date = date_range
        df_filtered = df_enso.loc[str(start_date):str(end_date)]
    else:
        df_filtered = df_enso.copy()
    
    df_reframed = series_to_supervised(df_filtered['ONI'], n_in, n_out, n_features)
    
    n = df_reframed.shape[0]
    n_train, n_valid = int(0.8 * n), int(0.1 * n)
    df_test = df_reframed.values[n_train + n_valid:, :]
    
    x_test, y_test = df_test[:, :-n_out], df_test[:, -n_out:]
    
    x_scaler = MinMaxScaler(feature_range=(0,1))
    y_scaler = MinMaxScaler(feature_range=(0,1))
    
    x_scaler.fit(df_reframed.values[:, :-n_out])
    y_scaler.fit(df_reframed.values[:, -n_out:])
    
    x_test = x_scaler.transform(x_test)
    y_test = y_scaler.transform(y_test)
    
    x_test = x_test.reshape(x_test.shape[0], n_steps, n_features)
    
    y_hat = model.predict(x_test)
    y_hat = np.round(y_scaler.inverse_transform(y_hat), 1)
    
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


def create_prediction_plot(y_actual, y_predict, y_forecast):
    fig = go.Figure()

    # Check if necessary columns exist
    required_columns = [('Actual', y_actual), ('Predicted', y_predict), ('Forecast', y_forecast)]
    for col, df in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in input DataFrame.")

    # Add actual values - keep original green
    fig.add_trace(go.Scatter(
        x=y_actual.index,
        y=y_actual['Actual'],
        name='Actual',
        line=dict(color='green', width=2),
        customdata=y_actual.index,  # Pass date as custom data
        hovertemplate='Actual: %{y:.2f}<extra></extra>'  # Remove date from each trace
    ))

    # Add predicted values - keep original blue
    fig.add_trace(go.Scatter(
        x=y_predict.index,
        y=y_predict['Predicted'],
        name='Predicted',
        line=dict(color='blue', width=2),  # Keep original blue
        customdata=y_predict.index,
        hovertemplate='Predicted: %{y:.2f}<extra></extra>'  # Remove duplicate date
    ))

    # Add forecast values - keep original red
    fig.add_trace(go.Scatter(
        x=y_forecast.index,
        y=y_forecast['Forecast'],
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),  # Keep original red dash
        customdata=y_forecast.index,
        hovertemplate='Forecast: %{y:.2f}<extra></extra>'  # Remove duplicate date
    ))

    fig.update_layout(
        title='ONI Prediction and Forecast',
        xaxis_title='Years',
        yaxis_title='ONI',
        template='plotly_white',
        hovermode='x unified',  # Keep hovermode unified
        title_font=dict(color='#2e7d32'),  # Green title ONLY
    )

    return fig


def create_enso_oni_plot(df_enso, date_range=None):
    # Filter by date range if provided
    if date_range:
        start_date, end_date = date_range
        df_filtered = df_enso.loc[str(start_date):str(end_date)]
    else:
        df_filtered = df_enso.copy()
    
    fig = go.Figure()
    
    # Define ENSO phase boundaries
    enso_phases = [
        (2.0, 3.0, "Very Strong El Niño", "red", 0.1),
        (1.5, 2.0, "Strong El Niño", "red", 0.08),
        (1.0, 1.5, "Moderate El Niño", "red", 0.06),
        (0.5, 1.0, "Weak El Niño", "red", 0.04),
        (-0.5, 0.5, "Neutral", "gray", 0.04),
        (-1.0, -0.5, "Weak La Niña", "blue", 0.04),
        (-1.5, -1.0, "Moderate La Niña", "blue", 0.06),
        (-2.0, -1.5, "Strong La Niña", "blue", 0.08),
        (-3.0, -2.0, "Very Strong La Niña", "blue", 0.1)
    ]
    
    annotations = []
    
    # Add horizontal ENSO phase regions
    for y0, y1, label, color, opacity in enso_phases:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=opacity, line_width=0)
        
        # Position labels to the right outside the graph
        annotations.append(dict(
            x=df_filtered.index[-1],  # Align at the last year on the right
            y=(y0 + y1) / 2,  # Centered in the middle of each phase
            text=label,
            showarrow=False,
            xanchor="left",  # Place the text outside the graph
            align='left',
            font=dict(size=12, color=color)
        ))

    # Add ONI line - keep original black
    fig.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered['ONI'],
        name='ONI',
        line=dict(color='black', width=2)  # Keep original black
    ))

    # Update layout
    fig.update_layout(
        title='ENSO and ONI Relation',
        xaxis_title='Years',
        yaxis_title='ONI',
        showlegend=True,
        template='plotly_white',
        annotations=annotations,
        yaxis=dict(
            tickmode='array',
            tickvals=[-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5],  # Custom tick values
            ticktext=["-2.5", "-2", "-1.5", "-1", "-0.5", "0", "0.5", "1", "1.5", "2", "2.5"],  # Custom labels
        ),
        title_font=dict(color='#2e7d32'),  # Green title ONLY
    )

    return fig

def main():
    st.title("🌊 ENSO Prediction Dashboard")
    
    with st.spinner('Loading data and model...'):
        df_enso, model = load_data_and_model()
    
    if df_enso is None or model is None:
        return
    
    # Add date range selector and summary statistics
    date_range = add_date_range_selector(df_enso)
    add_summary_statistics(df_enso)
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    plot_type = st.sidebar.selectbox(
        "Select Plot Type",
        ["All Plots", "Predictions and Forecast", "ENSO-ONI Relationship"]
    )
    
    # Add confidence interval control
    show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True)
    if show_confidence:
        confidence_level = st.sidebar.slider(
            "Confidence Level (σ)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with st.spinner('Processing data...'):
        y_actual, y_predict, y_forecast = process_data(df_enso, model, date_range)
    
    # 3-Month Forecast with keeping original colors
    st.sidebar.header("3-Month Forecast")
    for date, value in y_forecast.iterrows():
        phase = "El Niño" if value['Forecast'] > 0.5 else "La Niña" if value['Forecast'] < -0.5 else "Neutral"
        color = "red" if phase == "El Niño" else "blue" if phase == "La Niña" else "gray"
        delta_color = f"<span style='color:{color}'>{phase}</span>"
        st.sidebar.markdown(
            f"**{date.strftime('%Y-%m')}**: {value['Forecast']:.2f} - {delta_color}",
            unsafe_allow_html=True
        )
    
    # Calculate and display metrics
    st.header("Model Performance Metrics")
    metrics = calculate_metrics(y_actual, y_predict)
    cols = st.columns(3)
    for i, (metric, value) in enumerate(metrics.items()):
        cols[i].metric(metric, f"{value:.4f}")
    
    # Display plots
    if plot_type in ["All Plots", "Predictions and Forecast"]:
        st.header("ONI Predictions and Forecast")
        fig = create_prediction_plot(y_actual, y_predict, y_forecast)
        if show_confidence:
            add_confidence_intervals(fig, y_forecast, confidence_level)
        st.plotly_chart(fig, use_container_width=True)
        add_plot_export(fig)
    
    if plot_type in ["All Plots", "ENSO-ONI Relationship"]:
        st.header("ENSO-ONI Relationship")
        fig = create_enso_oni_plot(df_enso, date_range)
        st.plotly_chart(fig, use_container_width=True)
        add_plot_export(fig, key_suffix="enso_oni")
    
    # Add download button for data
    st.header("Download Data")
    add_download_button(y_actual, y_predict, y_forecast)

if __name__ == "__main__":
    main()
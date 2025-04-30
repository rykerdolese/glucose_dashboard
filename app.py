import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from darts import TimeSeries
from scipy import signal
from pykalman import KalmanFilter
import io
from sklearn.preprocessing import MinMaxScaler
from darts.models import XGBModel, BlockRNNModel, NaiveSeasonal


# Define relative paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # Current directory where the CSV is
MODEL_DIR = os.path.join(DATA_DIR, "saved_models")

from dash import Dash, html

app = Dash(__name__)

#Initialize the app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                assets_folder='assets')  # This is critical

server = app.server

def create_default_glucose_fig():
    fig = go.Figure()
    fig.update_layout(
        title="Glucose Predictions (Loading...)",
        xaxis_title="Time",
        yaxis_title="Glucose Level (mg/dL)",
        height=600,
        annotations=[dict(
            text="Loading data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            font=dict(size=16),
            showarrow=False
        )]
    )
    return fig

def create_default_gki_fig():
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source="https://via.placeholder.com/400x300.png",
            xref="paper", yref="paper",
            x=0, y=1,
            sizex=1, sizey=1,
            xanchor="left", yanchor="top",
            opacity=0.4,
            layer="below"
        )
    )
    fig.update_layout(
        title="GKI Analysis (Loading...)",
        xaxis_title="Time",
        yaxis_title="GKI Value",
        height=400
    )
    return fig

def create_default_future_fig():
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source="https://via.placeholder.com/400x300.png",
            xref="paper", yref="paper",
            x=0, y=1,
            sizex=1, sizey=1,
            xanchor="left", yanchor="top",
            opacity=0.4,
            layer="below"
        )
    )
    fig.update_layout(
        title="Future Glucose Prediction (Loading...)",
        xaxis_title="Time",
        yaxis_title="Predicted Glucose (mg/dL)",
        height=600
    )
    return fig




# Define optimal weights for ensemble model (example values - should be replaced with your actual weights)
OPTIMAL_WEIGHTS = {
    1: 0,
    2: 0.1,
    3: 0.1,
    4: 0.1,
    5: 0.2,
    6: 0.3,
    7: 0.3,
    8: 0.3,
    9: 0.4,
    10: 0.4,
    11: 0.5,
    12: 0.5,
    13: 0.5,
    14: 0.6,
    15: 0.6,
    16: 0.7,
    17: 0.7,
    18: 0.7,
    19: 0.7,
    20: 0.7,
    21: 0.7,
    22: 0.7,
    23: 0.7,
    24: 0.7,
}

def preprocess(df, source_start='2023-12-17 15:10:00-05:00', source_end='2023-12-22 23:45:00-05:00',
               target_start='2024-08-09 15:10:00-05:00', target_end='2024-08-14 23:45:00-05:00'):
    """
    Preprocesses the glucose reading data by copying values from a source date range 
    (2023-12-17 to 2023-12-22) to a target date range (2024-08-09 to 2024-08-14),
    with slight added noise, and marks interpolated values.
    """
    df = df.copy()

    source_start = pd.to_datetime(source_start, errors='coerce', utc=True).tz_convert('US/Central')
    source_end = pd.to_datetime(source_end, errors='coerce', utc=True).tz_convert('US/Central')
    target_start = pd.to_datetime(target_start, errors='coerce', utc=True).tz_convert('US/Central')
    target_end = pd.to_datetime(target_end, errors='coerce', utc=True).tz_convert('US/Central')

    # Convert time_cst to datetime and localize to US/Central timezone
    df['time_cst'] = pd.to_datetime(df['time_cst'], errors='coerce', utc=True).dt.tz_convert('US/Central')

    # Define the source and target masks
    source_mask = (df['time_cst'] >= source_start) & (df['time_cst'] <= source_end)
    target_mask = (df['time_cst'] >= target_start) & (df['time_cst'] <= target_end)

    # Extract source glucose values (from 2023-12-17 to 2023-12-22)
    source_glucose = df.loc[source_mask, 'Glucose Reading (mg/dL)'].values

    # If source and target ranges have the same length, copy values with noise
    if len(source_glucose) == target_mask.sum():
        # Apply slight noise (mean=0, std=2)
        noise = np.random.normal(0, 2, size=len(source_glucose))
        df.loc[target_mask, 'Glucose Reading (mg/dL)'] = source_glucose + noise
    else:
        raise ValueError("Source and target date ranges must have the same number of readings.")
    
    # Mark interpolated values
    df['Interpolated Glucose'] = target_mask.astype(int)
    df = df.astype({col: int for col in df.select_dtypes(include=['bool']).columns})

    # Remove timezone information from 'time_cst'
    df['time_cst'] = pd.to_datetime(df['time_cst'], errors='coerce', utc=True).dt.tz_localize(None)

    df = df.drop_duplicates(subset=['time_cst'], keep='first')
    df['time_cst'] = pd.to_datetime(df['time_cst'], errors='coerce', utc=True).dt.tz_localize(None)

    return df, target_mask

def apply_kalman_smoother(df, glucose_col="Glucose Reading (mg/dL)"):
    """Apply an adaptive Kalman smoother to glucose data, adjusting Q dynamically based on glucose variability."""
    df = df.copy()
    glucose_values = df[glucose_col].values.reshape(-1, 1)

    # Static parameters
    A = np.array([[1, 1], [0, 1]])  # State transition model
    H = np.array([[1, 0]])  # Observation model
    R = 0.2  # Measurement noise covariance (based on CGM accuracy)

    # Initial values
    initial_state_mean = [glucose_values[0, 0], 0]
    initial_state_covariance = np.array([[1, 0], [0, 0.1]])

    # Define baseline Q values
    Q_low = np.array([[0.02, 0], [0, 0.0015]])  # When glucose is stable
    Q_high = np.array([[0.1, 0], [0, 0.01]])   # When glucose is fluctuating

    # Compute glucose variability (ΔG_t = |G_t - G_t-1|)
    glucose_diff = np.abs(np.diff(glucose_values.flatten(), prepend=glucose_values[0]))

    # Normalize variability to scale Q dynamically
    variability_scale = np.clip(glucose_diff / np.max(glucose_diff), 0, 1)  # Scale between 0 and 1

    # Interpolate Q values dynamically
    Q_dynamic = np.array([
        (1 - scale) * Q_low + scale * Q_high for scale in variability_scale
    ])

    # Initialize Kalman smoother
    kf = KalmanFilter(
        transition_matrices=A, observation_matrices=H,
        observation_covariance=R,
        initial_state_mean=initial_state_mean, initial_state_covariance=initial_state_covariance
    )

    # Apply adaptive smoothing step by step
    smoothed_means = []
    state_mean = initial_state_mean
    state_cov = initial_state_covariance

    for t in range(len(glucose_values)):
        # Use dynamically selected Q at each time step
        kf.transition_covariance = Q_dynamic[t]

        # Kalman update step
        state_mean, state_cov = kf.filter_update(
            state_mean, state_cov, glucose_values[t]
        )

        smoothed_means.append(state_mean[0]) 

    df["Smoothed Glucose"] = smoothed_means
    return df

def load_models():
    """Load the trained XGBoost and LSTM models for ensemble prediction"""
    try:
        # Load XGB model
        xgb_model = XGBModel.load("xgb_smoothed_glucose_forecast_quantiles84_44_v5.pkl")
        
        # Load LSTM model
        lstm_model = BlockRNNModel(
            model="LSTM", input_chunk_length=45, output_chunk_length=24,
            hidden_dim=12, n_rnn_layers=1, batch_size=32, 
            dropout=0.18, n_epochs=50, optimizer_kwargs={"lr": 0.000653},
            random_state=42, force_reset=True
        )
        # Load with dummy training
        dummy_series = TimeSeries.from_values(np.zeros((100, 1))).astype(np.float32)
        lstm_model.fit(dummy_series, epochs=1, verbose=True)
        lstm_model.load_weights("lstm_model1_12_v2")
        
        # Load naive model
        naive_model = NaiveSeasonal(K=1)
        
        print("✔️ Models loaded successfully")
        return xgb_model, lstm_model, naive_model
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None

def load_and_preprocess_data():
    """Loads and preprocesses glucose data using the improved functions."""
    try:
        # Read the CSV file
        #df = pd.read_csv(os.path.join(DATA_DIR, 'full_gluket_cleaned_data.csv'))
        df = pd.read_csv('full_gluket_cleaned_data_small.csv')
        # Apply preprocessing and smoothing
        df, _ = preprocess(df)
        df = apply_kalman_smoother(df)
        
        # Convert boolean columns to integers for the model
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)
        
        # Add hour, month, day_of_week features
        df['hour'] = df['time_cst'].dt.hour
        df['month'] = df['time_cst'].dt.month
        df['day_of_week'] = df['time_cst'].dt.dayofweek
        
        # Calculate GKI
        df['GKI'] = df['Glucose Reading (mg/dL)'] / (df['Ketone Reading (mmol)'] * 18)
        
        print("✅ Data loaded and preprocessed successfully!")
        print(f"Time range: {df['time_cst'].min()} to {df['time_cst'].max()}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Sample time intervals for future prediction (5-minute intervals)
FUTURE_INTERVALS = 24  # 3 hours in 5-minute intervals

model_selector_dropdown = dbc.Row([
    dbc.Col([
        html.Label("Select Forecasting Model:", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id='model-selector',
            options=[
                {'label': 'XGBoost', 'value': 'xgboost'},
                {'label': 'LSTM', 'value': 'lstm'},
                {'label': 'Ensemble', 'value': 'ensemble'}
            ],
            value='ensemble',  # Default selection
            clearable=False
        )
    ], width=6, className="mb-4")
])


# Create layout
app.layout = dbc.Container([
    
    dbc.Row([
        dbc.Col([
            html.H1("Glucose Prediction Dashboard", className="text-center mb-4 mt-4"),
            dcc.Store(id='model-loaded-flag'),
    dcc.Store(id='processed-data'),

    dcc.Loading(
    id="loading-indicator",
    type="circle",
    #fullscreen=True,
    children=html.Div([
        html.Div(id="loading-message"),
        html.Div(id="model-loaded-flag")
    ])
),

   
            html.Hr()
        ], width=12)
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="Historical Data Analysis", children=[
            model_selector_dropdown,
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Select Date Range"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Start Date:"),
                                    dcc.DatePickerSingle(
                                        id='start-date-picker',
                                        min_date_allowed=datetime(2025, 1, 1),
                                        max_date_allowed=datetime(2025, 1,12),
                                        initial_visible_month=datetime(2025, 1, 1),
                                        date=datetime(2025, 1, 1)
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("End Date:"),
                                    dcc.DatePickerSingle(
                                        id='end-date-picker',
                                        min_date_allowed=datetime(2025, 1, 1),
                                        max_date_allowed=datetime(2025, 1,12),
                                        initial_visible_month=datetime(2025, 1, 1),
                                        date=datetime(2025, 1, 2)
                                    ),
                                ], width=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Generate Analysis", id="generate-button", color="primary", className="mt-3"),
                                    html.Div(id="loading-output")
                                ], width=12, className="text-center")
                            ]),
                        ])
                    ], className="mb-4")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Glucose Predictions"),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-graph",
                                type="default",
                                children=[
                                    dcc.Graph(id="prediction-graph", figure=create_default_glucose_fig(), style={"height": "600px"})
                                ]
                            )
                        ])
                    ])
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("GKI Analysis"),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-gki-graph",
                                type="default",
                                children=[
                                    dcc.Graph(id="gki-graph", figure=create_default_gki_fig(), style={"height": "400px"})
                                ]
                            )
                        ])
                    ], className="mt-4")
                ], width=12)
            ]),
        ]),
        
        dbc.Tab(label="Future Prediction", children=[
            model_selector_dropdown,
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Input Future Values for Prediction"),
                        dbc.CardBody([
                            html.P("Select a starting point and input your parameters to predict future glucose values:"),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Starting Reference Date:"),
                                    dcc.DatePickerSingle(
                                        id='future-start-date',
                                        min_date_allowed=datetime(2025, 1, 2),
                                        max_date_allowed=datetime(2025, 1,12),
                                        initial_visible_month=datetime(2025, 1, 1),
                                        date=datetime(2025, 1, 8)
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Starting Time:"),
                                    dcc.Dropdown(
                                        id='future-start-time',
                                        options=[
                                            {'label': f"{h:02d}:{m:02d}", 'value': f"{h:02d}:{m:02d}"} 
                                            for h in range(24) for m in [0, 15, 30, 45]
                                        ],
                                        value="08:00",
                                        clearable=False
                                    ),
                                ], width=6),
                            ], className="mb-3"),
                            html.Div(id='values-source-indicator', className="mb-3", style={"fontStyle": "italic", "color": "#6c757d"}),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Current Glucose Reading (mg/dL):"),
                                    dbc.Input(
                                        id='current-glucose',
                                        type='number',
                                        min=40,
                                        max=400,
                                        step=1,
                                        value=100
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Current Ketone Reading (mmol):"),
                                    dbc.Input(
                                        id='current-ketone',
                                        type='number',
                                        min=0.1,
                                        max=10,
                                        step=0.1,
                                        value=0.5
                                    ),
                                ], width=6),
                            ], className="mb-3"),
                            
                            html.H5("Meal Information", className="mt-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Energy (kcal):"),
                                    dbc.Input(
                                        id='future-energy',
                                        type='number',
                                        min=0,
                                        step=10,
                                        value=0
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Carbohydrates (g):"),
                                    dbc.Input(
                                        id='future-carbs',
                                        type='number',
                                        min=0,
                                        step=1,
                                        value=0
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Protein (g):"),
                                    dbc.Input(
                                        id='future-protein',
                                        type='number',
                                        min=0,
                                        step=1,
                                        value=0
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Fat (g):"),
                                    dbc.Input(
                                        id='future-fat',
                                        type='number',
                                        min=0,
                                        step=1,
                                        value=0
                                    ),
                                ], width=3),
                            ], className="mb-3"),
                            
                            html.H5("Exercise Information", className="mt-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("In Exercise:"),
                                    dcc.RadioItems(
                                        id='future-in-exercise',
                                        options=[
                                            {'label': 'Yes', 'value': 1},
                                            {'label': 'No', 'value': 0}
                                        ],
                                        value=0,
                                        inline=True
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Running:"),
                                    dcc.RadioItems(
                                        id='future-running',
                                        options=[
                                            {'label': 'Yes', 'value': 1},
                                            {'label': 'No', 'value': 0}
                                        ],
                                        value=0,
                                        inline=True
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Strength Training:"),
                                    dcc.RadioItems(
                                        id='future-strength',
                                        options=[
                                            {'label': 'Yes', 'value': 1},
                                            {'label': 'No', 'value': 0}
                                        ],
                                        value=0,
                                        inline=True
                                    ),
                                ], width=3),
                                dbc.Col([
                                    html.Label("Strenuous Exercise:"),
                                    dcc.RadioItems(
                                        id='future-strenuous',
                                        options=[
                                            {'label': 'Yes', 'value': 1},
                                            {'label': 'No', 'value': 0}
                                        ],
                                        value=0,
                                        inline=True
                                    ),
                                ], width=3),
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Generate Future Prediction", id="future-predict-button", color="success", className="mt-3 w-100"),
                                ], width=12),
                            ]),
                            
                            html.Div(id="future-prediction-status", className="mt-2"),
                        ])
                    ], className="mb-4"),
                ], width=12),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Future Glucose Prediction"),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-future-graph",
                                type="default",
                                children=[
                                    dcc.Graph(id="future-prediction-graph", figure=create_default_future_fig(), style={"height": "600px"})
                                ]
                            )
                        ])
                    ])
                ], width=12)
            ]),
        ]),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Information"),
                dbc.CardBody([
                    html.P("This dashboard uses an ensemble model combining XGBoost and LSTM models trained to predict glucose levels."),
                    html.P("The model forecasts glucose levels from 5 minutes to 2 hours into the future with horizon-specific weights."),
                    html.P("Data is pre-processed with adaptive Kalman smoothing to reduce noise while preserving important trends."),
                    html.P("GKI (Glucose-Ketone Index) is displayed to show the relationship between glucose and ketone levels.")
                ])
            ], className="mt-4 mb-4")
        ], width=12)
    ]),
    
    # Store components for data
     dcc.Store(id='processed-data'),
     dcc.Store(id='model-loaded-flag')
], fluid=True)

from dash import ctx

@app.callback(
    Output('model-loaded-flag', 'data'),
    #Output('loading-status', 'children'),
    Input('model-loaded-flag', 'data')
)
def load_model_on_startup(current_value):
    if current_value is None:
        global xgb_model, lstm_model, naive_model
        return_msg = "Loading models..."
        xgb_model, lstm_model, naive_model = load_models()
        loaded = xgb_model is not None and lstm_model is not None
        return {'loaded': loaded}, "Models loaded." if loaded else "Model loading failed."
    return current_value, "Models already loaded."


@app.callback(
    Output('processed-data', 'data'),
    Output('loading-message', 'children'),
    #Output('loading-indicator', 'children'),
    Input('model-loaded-flag', 'data')
)

def load_data_on_startup(model_loaded):
    if model_loaded and model_loaded[0].get('loaded', False):
        status_msg = "Loading and preprocessing data..."
        df = load_and_preprocess_data()
        if df is not None:
            df['time_cst'] = df['time_cst'].astype(str)
            return df.to_json(date_format='iso', orient='split'), "Data loaded successfully."
        return None, "Data loading failed."
    return None, "Waiting for models to load..."

@app.callback(
    [Output('prediction-graph', 'figure'),
     Output('gki-graph', 'figure'),
     Output('loading-output', 'children')],
    [Input('generate-button', 'n_clicks'),
     Input('model-selector', 'value')],  # <-- new input
    [State('processed-data', 'data'),
     State('start-date-picker', 'date'),
     State('end-date-picker', 'date'),
     State('model-loaded-flag', 'data')]
)
def update_graphs(n_clicks, model_choice, data_json, start_date, end_date, model_loaded):
    if n_clicks is None or data_json is None:
        return {}, {}, ""
    
    try:
        OPTIMAL_WEIGHTS = {
            1: 0,
            2: 0.1,
            3: 0.1,
            4: 0.1,
            5: 0.2,
            6: 0.3,
            7: 0.3,
            8: 0.3,
            9: 0.4,
            10: 0.4,
            11: 0.5,
            12: 0.5,
            13: 0.5,
            14: 0.6,
            15: 0.6,
            16: 0.7,
            17: 0.7,
            18: 0.7,
            19: 0.7,
            20: 0.7,
            21: 0.7,
            22: 0.7,
            23: 0.7,
            24: 0.7,
        }
        # Convert back from JSON to DataFrame
        df = pd.read_json(io.StringIO(data_json), orient='split')
        # Important: Convert time_cst to datetime *without timezone* to avoid comparison errors
        df['time_cst'] = pd.to_datetime(df['time_cst'], utc=True).dt.tz_localize(None)
        
        # Convert date strings to datetime objects - also without timezone
        start_date = pd.to_datetime(start_date).replace(tzinfo=None)
        end_date = pd.to_datetime(end_date).replace(tzinfo=None) + timedelta(days=1)  # Include the end date
        
        # Filter data based on selected date range
        mask = (df['time_cst'] >= start_date) & (df['time_cst'] <= end_date)
        df_filtered = df[mask]
        
        if df_filtered.empty:
            return {}, {}, "No data available for the selected date range"
        
        # Create glucose prediction figure
        glucose_fig = make_subplots(specs=[[{"secondary_y": False}]])
        
     
        
        # Add smoothed glucose data
        # glucose_fig.add_trace(
        #     go.Scatter(
        #         x=df_filtered['time_cst'] + pd.Timedelta(hours=3),  # Adjust for timezone
        #         y=df_filtered['Smoothed Glucose'],
        #         mode='lines',
        #         name='Smoothed Glucose',
        #         line=dict(color='purple', dash='dot')
        #     )
        # )
        

        # Generate predictions if model is loaded
        if model_loaded and model_loaded[0].get('loaded', False) and xgb_model is not None and lstm_model is not None:
            try:
                # Create time series objects
                target_series = TimeSeries.from_dataframe(
                    df_filtered, 
                    time_col='time_cst', 
                    value_cols="Smoothed Glucose",
                    freq='5min'
                )

                covariates_cols = [
                    'Ketone Reading (mmol)', 'measurement_time', 'Energy (kcal)', 'Carbohydrates (g)', 
                    'Protein (g)', 'Fat (g)', 'hour', 'month', 'day_of_week', 'In Exercise', 
                    'Running', 'Strength Training', 'Strenuous Exercise', 'Interpolated Glucose'
                ]
                covariates = TimeSeries.from_dataframe(
                    df_filtered, 
                    time_col='time_cst', 
                    value_cols=covariates_cols,
                    freq='5min'
                )
                
                # Read in first dataset: training data, validation data
                df_whole = pd.read_csv('full_gluket_cleaned_data_small.csv')
                df_whole, mask = preprocess(df_whole)
                df_whole = apply_kalman_smoother(df_whole)
                

                # Create darts time series objects
                train_target = TimeSeries.from_dataframe(df_whole, time_col='time_cst', value_cols="Smoothed Glucose").astype(np.float32)
                train_covs = TimeSeries.from_dataframe(df_whole, time_col='time_cst', value_cols=[
                    'Ketone Reading (mmol)', 'measurement_time', 'Energy (kcal)', 'Carbohydrates (g)', 'Protein (g)', 'Fat (g)',
                    'hour', 'month', 'day_of_week', 'In Exercise', 'Running', 'Strength Training', 'Strenuous Exercise', 'Interpolated Glucose'
                ]).astype(np.float32)
                

                # Scaling for LSTM
                scaler = MinMaxScaler()
                scaler_glucose = MinMaxScaler()

                scaler_glucose.fit(df_whole[['Smoothed Glucose']])
                df_whole['Smoothed Glucose'] = scaler_glucose.transform(df_whole[['Smoothed Glucose']]).astype(np.float32)

                scaled_features = [
                    'Ketone Reading (mmol)', 'measurement_time', 'Energy (kcal)', 'Carbohydrates (g)', 
                    'Protein (g)', 'Fat (g)', "Smoothed Glucose", 'hour', 'month', 'day_of_week'
                ]

                df_whole[scaled_features] = scaler.fit(df_whole[scaled_features])


                combined_df = df_filtered
                combined_df[scaled_features] = scaler.transform(combined_df[scaled_features]).astype(np.float32)
                combined_df['Smoothed Glucose'] = scaler_glucose.transform(combined_df[['Smoothed Glucose']]).astype(np.float32)

                # Create scaled time series objects for LSTM
                target_lstm = TimeSeries.from_dataframe(combined_df, time_col='time_cst', value_cols="Smoothed Glucose").astype(np.float32)
                covs_lstm = TimeSeries.from_dataframe(combined_df, time_col='time_cst', value_cols=[
                    'Ketone Reading (mmol)', 'measurement_time', 'Energy (kcal)', 'Carbohydrates (g)', 'Protein (g)', 'Fat (g)',
                    'hour', 'month', 'day_of_week', 'In Exercise', 'Running', 'Strength Training', 'Strenuous Exercise', 'Interpolated Glucose'
                ]).astype(np.float32)   

                glucose_fig.add_trace(
                go.Scatter(
                    x=target_series.time_index,  # Adjust for timezone
                    y=target_series.values().flatten(),
                    mode='lines',
                    name='Smoothed Glucose',
                    line=dict(color='purple', dash='dot')
                )
                )

            except Exception as e:
                print(f"Error generating predictions: {e}")
                import traceback
                traceback.print_exc()
            
        
        

                # Generate predictions for different horizons
                # Loop through forecast horizons — for now, only 45-minute (horizon=9)
        for horizon, color, name in [(2, 'black', '15-minute'), (5, 'red', '30-minute'), (11, 'orange', '1-hour')]:
            try:
                
                if model_choice == 'xgboost':
                    XGB_forecast_mean = xgb_model.historical_forecasts(
                        series=target_series,
                        past_covariates=covariates,
                        forecast_horizon=horizon,
                        stride=1,
                        retrain=False,
                    )
                    xgb_forecast_time = XGB_forecast_mean.time_index#.pd_series().index
                    XGB_forecast_mean = XGB_forecast_mean.values().flatten()
                    
                    print(xgb_forecast_time)
                    print(XGB_forecast_mean)
                    glucose_fig.add_trace(go.Scatter(
                    x=xgb_forecast_time,
                    y=XGB_forecast_mean,
                    mode='lines',
                    name=f'{name} Forecast (XGB)',
                    line=dict(color=color)
                    
                ))
                    
                    
                elif model_choice == 'lstm':
                    import os
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                    import torch

                    # Force CPU usage
                    torch_device = torch.device("cpu")
                    lstm_model.model = lstm_model.model.to(torch_device)
                    LSTM_forecast_mean_scaled = lstm_model.historical_forecasts(
                        series=target_lstm,
                        past_covariates=covs_lstm,
                        forecast_horizon=horizon,
                        stride=1,
                        retrain=False
                    )
                    lstm_values_scaled = LSTM_forecast_mean_scaled.values().reshape(-1, 1)
                    LSTM_forecast_mean = scaler_glucose.inverse_transform(lstm_values_scaled).flatten()
                    lstm_forecast_time = LSTM_forecast_mean_scaled.time_index#.pd_series().index
                    glucose_fig.add_trace(go.Scatter(
                    x=lstm_forecast_time,
                    y=LSTM_forecast_mean,
                    mode='lines',
                    name=f'{name} Forecast (LSTM)',
                    line=dict(color=color)
                ))

                elif model_choice == 'ensemble':
                    import os
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                    import torch

                    # Force CPU usage
                    torch_device = torch.device("cpu")
                    lstm_model.model = lstm_model.model.to(torch_device)
                    
                    # Get LSTM forecast (scaled)
                    LSTM_forecast_mean_scaled = lstm_model.historical_forecasts(
                        series=target_lstm,
                        past_covariates=covs_lstm,
                        forecast_horizon=horizon,
                        stride=1,
                        retrain=False
                    )
                    
                    # Get XGB forecast
                    XGB_forecast_mean = xgb_model.historical_forecasts(
                        series=target_series,
                        past_covariates=covariates,
                        forecast_horizon=horizon,
                        stride=1,
                        retrain=False,
                    )

                    # Inverse transform LSTM forecast to original scale
                    lstm_values_scaled = LSTM_forecast_mean_scaled.values().reshape(-1, 1)
                    LSTM_forecast_mean = scaler_glucose.inverse_transform(lstm_values_scaled).flatten()
                    
                    # Get XGB values
                    xgb_values = XGB_forecast_mean.values().flatten()
                    
                    # Get time index (use XGB's time index as reference)
                    xgb_forecast_time = XGB_forecast_mean.time_index
                    
                    # Make sure we have matching lengths
                    min_length = min(len(xgb_values), len(LSTM_forecast_mean))
                    xgb_values = xgb_values#[:min_length]
                    LSTM_forecast_mean = LSTM_forecast_mean[39:]#[:min_length]
                    
                    # Create ensemble prediction
                    weight = OPTIMAL_WEIGHTS.get(horizon, 0.5)  # Default to 0.5 if not found
                    ensemble_pred = (weight * xgb_values) + ((1 - weight) * LSTM_forecast_mean)

                    print(ensemble_pred)
                    glucose_fig.add_trace(go.Scatter(
                        x=xgb_forecast_time,#[:min_length],
                        y=ensemble_pred,
                        mode='lines',
                        name=f'{name} Forecast (Ensemble)',
                        line=dict(color=color)
                    ))
                
            except Exception as e:
                print(f"Error generating predictions for {name}: {e}")
                import traceback
                traceback.print_exc()
        
       
        
        # Update layout
        glucose_fig.update_layout(
            title='Glucose Predictions',
            xaxis_title='Time',
            yaxis_title='Glucose Level (mg/dL)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600,
            hovermode="x unified"
        )
       
        
        # Create GKI graph
        gki_fig = go.Figure()
        
        # Add GKI data
        gki_fig.add_trace(
            go.Scatter(
                x=df_filtered['time_cst'],
                y=df_filtered['GKI'],
                mode='lines',
                name='GKI',
                line=dict(color='teal')
            )
        )
        
        # Add reference lines for ketosis states
        gki_fig.add_shape(
            type="line",
            x0=min(df_filtered['time_cst']),
            y0=9,
            x1=max(df_filtered['time_cst']),
            y1=9,
            line=dict(color="green", width=1, dash="dash"),
        )
        
        gki_fig.add_shape(
            type="line",
            x0=min(df_filtered['time_cst']),
            y0=6,
            x1=max(df_filtered['time_cst']),
            y1=6,
            line=dict(color="blue", width=1, dash="dash"),
        )
        
        gki_fig.add_shape(
            type="line",
            x0=min(df_filtered['time_cst']),
            y0=3,
            x1=max(df_filtered['time_cst']),
            y1=3,
            line=dict(color="purple", width=1, dash="dash"),
        )
        
        # Update GKI layout
        gki_fig.update_layout(
            title='Glucose-Ketone Index (GKI)',
            xaxis_title='Time',
            yaxis_title='GKI Value',
            height=400,
            hovermode="x unified"
        )
        
        # Add annotations for ketosis states
        gki_fig.add_annotation(
            x=min(df_filtered['time_cst']),
            y=9,
            text="Mild Ketosis",
            showarrow=False,
            yshift=10,
            xshift=5
        )
        
        gki_fig.add_annotation(
            x=min(df_filtered['time_cst']),
            y=6,
            text="Moderate Ketosis",
            showarrow=False,
            yshift=10,
            xshift=5
        )
        
        gki_fig.add_annotation(
            x=min(df_filtered['time_cst']),
            y=3,
            text="Deep Ketosis",
            showarrow=False,
            yshift=10,
            xshift=5
        )
        
        return glucose_fig, gki_fig, "Analysis generated successfully!"
    
    except Exception as e:
        print(f"Error generating analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}, f"Error: {str(e)}"
    
@app.callback(
    Output('values-source-indicator', 'children'),
    [Input('future-start-date', 'date'),
     Input('future-start-time', 'value')]
)
def update_values_indicator(start_date, start_time):
    if start_date and start_time:
        return f"Values loaded from historical data at {start_time} on {start_date}"
    return "Using default values - select a date/time to load historical data"

def update_input_values(start_date, start_time, data_json):
    """Update input fields based on selected date/time"""
    if not start_date or not start_time or not data_json:
        # Return default values if no data
        return 100, 0.5, 0, 0, 0, 0, 0, 0, 0, 0
    
    try:
        # Load and prepare data
        df = pd.read_json(io.StringIO(data_json), orient='split')
        df['time_cst'] = pd.to_datetime(df['time_cst'])
        
        # Create datetime object from inputs
        hour, minute = map(int, start_time.split(':'))
        selected_datetime = pd.to_datetime(start_date).replace(hour=hour, minute=minute)
        
        # Find the closest record to selected time
        time_diff = (df['time_cst'] - selected_datetime).abs()
        closest_idx = time_diff.idxmin()
        closest_row = df.iloc[closest_idx]
        
        # Return values from the closest record
        return (
            closest_row['Glucose Reading (mg/dL)'],
            closest_row['Ketone Reading (mmol)'],
            closest_row['Energy (kcal)'],
            closest_row['Carbohydrates (g)'],
            closest_row['Protein (g)'],
            closest_row['Fat (g)'],
            int(closest_row['In Exercise']),
            int(closest_row['Running']),
            int(closest_row['Strength Training']),
            int(closest_row['Strenuous Exercise'])
        )
    
    except Exception as e:
        print(f"Error updating input values: {e}")
        # Return default values if error occurs
        return 100, 0.5, 0, 0, 0, 0, 0, 0, 0, 0
    
@app.callback(
    [
        Output('current-glucose', 'value'),
        Output('current-ketone', 'value'),
        Output('future-energy', 'value'),
        Output('future-carbs', 'value'),
        Output('future-protein', 'value'),
        Output('future-fat', 'value'),
        Output('future-in-exercise', 'value'),
        Output('future-running', 'value'),
        Output('future-strength', 'value'),
        Output('future-strenuous', 'value')
    ],
    [Input('future-start-date', 'date'),
     Input('future-start-time', 'value')],
    [State('processed-data', 'data')]
)

def update_defaults_from_time(start_date, start_time, data_json):
    return update_input_values(start_date, start_time, data_json)
    
@app.callback(
    [Output('future-prediction-graph', 'figure'),
     Output('future-prediction-status', 'children')],
    [Input('future-predict-button', 'n_clicks'),
     Input('model-selector', 'value')],  # <-- NEW INPUT
    [State('processed-data', 'data'),
     State('model-loaded-flag', 'data'),
     State('future-start-date', 'date'),
     State('future-start-time', 'value'),
     State('current-glucose', 'value'),
     State('current-ketone', 'value'),
     State('future-energy', 'value'),
     State('future-carbs', 'value'),
     State('future-protein', 'value'),
     State('future-fat', 'value'),
     State('future-in-exercise', 'value'),
     State('future-running', 'value'),
     State('future-strength', 'value'),
     State('future-strenuous', 'value')]
)

def generate_future_prediction(n_clicks, model_choice, data_json, model_loaded, start_date, start_time, 
                              glucose, ketone, energy, carbs, protein, fat, 
                              in_exercise, running, strength, strenuous):
    
    OPTIMAL_WEIGHTS = {
    1: 0,
    2: 0.1,
    3: 0.1,
    4: 0.1,
    5: 0.2,
    6: 0.3,
    7: 0.3,
    8: 0.3,
    9: 0.4,
    10: 0.4,
    11: 0.5,
    12: 0.5,
    13: 0.5,
    14: 0.6,
    15: 0.6,
    16: 0.7,
    17: 0.7,
    18: 0.7,
    19: 0.7,
    20: 0.7,
    21: 0.7,
    22: 0.7,
    23: 0.7,
    24: 0.7,
    }
    
    if n_clicks is None or data_json is None:
        return {}, ""

    if model_loaded is None or not model_loaded[0].get('loaded', False) or xgb_model is None or lstm_model is None:
        return {}, "Model not loaded. Please try again."

    try:
        df = pd.read_json(io.StringIO(data_json), orient='split')
        df['time_cst'] = pd.to_datetime(df['time_cst'], utc=True).dt.tz_localize(None)
        import pytz
        from datetime import datetime

        

        hour, minute = map(int, start_time.split(':'))
       # 1. When creating from user input (naive datetime)
        start_datetime = pd.to_datetime(start_date).replace(hour=hour, minute=minute)

        # 2. Localize as US/Central (if coming from naive datetime)
        start_datetime = pytz.timezone('US/Central').localize(start_datetime)
        start_datetime = start_datetime.tz_localize(None)

        #start_datetime += timedelta(hours=1)
        # Create future dataframe with user inputs
        future_times = pd.date_range(
            start=start_datetime, 
            periods=1, 
            freq='5min'
        )
        
        # Create future dataframe with user inputs
        future_df = pd.DataFrame({
            'time_cst': future_times,
            'Glucose Reading (mg/dL)': glucose,
            'Ketone Reading (mmol)': ketone,
            'Energy (kcal)': energy,
            'Carbohydrates (g)': carbs,
            'Protein (g)': protein,
            'Fat (g)': fat,
            'In Exercise': in_exercise,
            'Running': running,
            'Strength Training': strength,
            'Strenuous Exercise': strenuous,
            'Interpolated Glucose': 0
        })
        
        # Add time features
        future_df['hour'] = future_df['time_cst'].dt.hour
        future_df['month'] = future_df['time_cst'].dt.month
        future_df['day_of_week'] = future_df['time_cst'].dt.dayofweek
        future_df['measurement_time'] = 0  # Assuming this is a feature
        future_df['Smoothed Glucose'] =  glucose  # Placeholder for smoothed glucose
        
        # Combine with historical data for context
        
        context_df = df[df['time_cst'] < start_datetime].tail(84)  # Use last 84 points (7 hours) as context
        combined_df = pd.concat([context_df, future_df], ignore_index=True)
        
        
        # Apply Kalman smoothing to the combined data
        #combined_df = apply_kalman_smoother(combined_df)
        
        # Create time series objects
        target_series = TimeSeries.from_dataframe(
            combined_df, 
            time_col='time_cst', 
            value_cols="Smoothed Glucose",
            freq='5min'
        )
        
        covariates_cols = [
            'Ketone Reading (mmol)', 'measurement_time', 'Energy (kcal)', 'Carbohydrates (g)', 
            'Protein (g)', 'Fat (g)', 'hour', 'month', 'day_of_week', 'In Exercise', 
            'Running', 'Strength Training', 'Strenuous Exercise', 'Interpolated Glucose'
        ]
        covariates = TimeSeries.from_dataframe(
            combined_df, 
            time_col='time_cst', 
            value_cols=covariates_cols,
            freq='5min'
        )

        def preprocess_xgb_lstm_model_data(data_dir, train_data_file='full_gluket_cleaned_data_small.csv', combined_df = combined_df):
            """
            Preprocesses and prepares model data for training and testing.
            
            This function:
            1. Reads and preprocesses training and test datasets
            2. Applies Kalman smoothing
            3. Creates time series objects for targets and covariates
            4. Scales data for LSTM models
            5. Returns all necessary data structures for model training and evaluation
            
            Args:
                data_dir (str): Directory containing the data files
                train_data_file (str): Filename for training data (default: 'full_gluket_cleaned_data.csv')
                test_data_file (str): Filename for test data (default: 'merged_data_v2.csv')
                test_start_date (str): Start datetime for test period (default: '2024-12-11 17:05:00-06:00')
                test_end_date (str): End datetime for test period (default: '2025-01-12 22:40:00-06:00')
            
            Returns:
                tuple: A tuple containing:
                    - train_target: TimeSeries for training target (Smoothed Glucose)
                    - train_covs: TimeSeries for training covariates
                    - test_target: TimeSeries for test target
                    - test_covs: TimeSeries for test covariates
                    - future_test_covs: TimeSeries for future test covariates
                    - future_train_covs: TimeSeries for future train covariates
                    - train_target_lstm: Scaled TimeSeries for LSTM training target
                    - train_covs_lstm: Scaled TimeSeries for LSTM training covariates
                    - test_target_lstm: Scaled TimeSeries for LSTM test target
                    - test_covs_lstm: Scaled TimeSeries for LSTM test covariates
                    - future_test_covs_lstm: Scaled TimeSeries for LSTM future test covariates
                    - future_train_covs_lstm: Scaled TimeSeries for LSTM future train covariates
                    - scaler: MinMaxScaler fit to training data
                    - scaler_glucose: MinMaxScaler fit to glucose values
            """
            # Read in first dataset: training data, validation data
            df_whole = pd.read_csv('full_gluket_cleaned_data_small.csv')
            df_whole, mask = preprocess(df_whole)
            df_whole = apply_kalman_smoother(df_whole)
            

            # Create darts time series objects
            train_target = TimeSeries.from_dataframe(df_whole, time_col='time_cst', value_cols="Smoothed Glucose").astype(np.float32)
            train_covs = TimeSeries.from_dataframe(df_whole, time_col='time_cst', value_cols=[
                'Ketone Reading (mmol)', 'measurement_time', 'Energy (kcal)', 'Carbohydrates (g)', 'Protein (g)', 'Fat (g)',
                'hour', 'month', 'day_of_week', 'In Exercise', 'Running', 'Strength Training', 'Strenuous Exercise', 'Interpolated Glucose'
            ]).astype(np.float32)
            

            # Scaling for LSTM
            scaler = MinMaxScaler()
            scaler_glucose = MinMaxScaler()

            scaler_glucose.fit(df_whole[['Smoothed Glucose']])
            df_whole['Smoothed Glucose'] = scaler_glucose.transform(df_whole[['Smoothed Glucose']]).astype(np.float32)

            scaled_features = [
                'Ketone Reading (mmol)', 'measurement_time', 'Energy (kcal)', 'Carbohydrates (g)', 
                'Protein (g)', 'Fat (g)', "Smoothed Glucose", 'hour', 'month', 'day_of_week'
            ]

            df_whole[scaled_features] = scaler.fit(df_whole[scaled_features])



            combined_df[scaled_features] = scaler.transform(combined_df[scaled_features]).astype(np.float32)
            combined_df['Smoothed Glucose'] = scaler_glucose.transform(combined_df[['Smoothed Glucose']]).astype(np.float32)

            # Create scaled time series objects for LSTM
            target_lstm = TimeSeries.from_dataframe(combined_df, time_col='time_cst', value_cols="Smoothed Glucose").astype(np.float32)
            covs_lstm = TimeSeries.from_dataframe(combined_df, time_col='time_cst', value_cols=[
                'Ketone Reading (mmol)', 'measurement_time', 'Energy (kcal)', 'Carbohydrates (g)', 'Protein (g)', 'Fat (g)',
                'hour', 'month', 'day_of_week', 'In Exercise', 'Running', 'Strength Training', 'Strenuous Exercise', 'Interpolated Glucose'
            ]).astype(np.float32)
            

            return (
                train_target, train_covs, scaler, scaler_glucose, target_lstm, covs_lstm
            )
        
        train_target, train_covs, scaler, scaler_glucose, target_lstm, covs_lstm = preprocess_xgb_lstm_model_data(
            DATA_DIR, combined_df=combined_df
        )

        
        prediction_times = []
        prediction_values = []
        confidence_lower = []
        confidence_upper = []

        
        def generate_prediction_figure(model_choice, xgb_model, lstm_model, target_series, covariates, 
                            target_lstm, covs_lstm, scaler_glucose, optimal_weights):
            """Generate prediction figure based on selected model type"""
            fig = go.Figure()
            
            # Get historical data for context
            prior_time = target_series[-45:].time_index
            prior_values = target_series[-45:].values().flatten()
            
            # Common figure elements
            def add_common_elements(fig, prediction_times, prediction_values, 
                                confidence_lower=None, confidence_upper=None):
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=prior_time,
                    y=prior_values,
                    mode='lines',
                    name='Historical Glucose',
                    line=dict(color='blue')
                ))
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=prediction_times,
                    y=prediction_values,
                    mode='lines+markers',
                    name='Predicted Glucose',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))
                
                # Add confidence interval if provided
                if confidence_lower is not None and confidence_upper is not None:
                    fig.add_trace(go.Scatter(
                        x=prediction_times,
                        y=confidence_lower,
                        line=dict(color='rgba(255,0,0,0.2)'),
                        mode='lines',
                        name='Lower CI',
                        hoverinfo='skip',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=prediction_times,
                        y=confidence_upper,
                        line=dict(color='rgba(255,0,0,0.2)'),
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        name='Confidence Interval',
                        hoverinfo='skip',
                        showlegend=False
                    ))
                
                
                
                # Update layout
                fig.update_layout(
                    title="Future Glucose Prediction",
                    xaxis_title="Time",
                    yaxis_title="Predicted Glucose (mg/dL)",
                    hovermode="x unified",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                
                
                return fig
            
            if model_choice == 'xgboost':
                # XGBoost prediction
                XGB_forecast_mean = xgb_model.predict(
                    series=target_series,
                    past_covariates=covariates,
                    n=24
                )
                
                XGB_forecast_conf = xgb_model.predict(
                    series=target_series,
                    past_covariates=covariates,
                    n=24,
                    num_samples=200
                )
                
                prediction_times = XGB_forecast_mean.time_index
                prediction_values = XGB_forecast_mean.values().flatten()
                confidence_lower = XGB_forecast_conf.quantile_timeseries(0.1).values().flatten()
                confidence_upper = XGB_forecast_conf.quantile_timeseries(0.9).values().flatten()
                
                return add_common_elements(fig, prediction_times, prediction_values, 
                                        confidence_lower, confidence_upper)
            
            elif model_choice == 'lstm':
                # LSTM prediction
                LSTM_forecast_mean_scaled = lstm_model.predict(
                    series=target_lstm[39:],
                    past_covariates=covs_lstm,
                    n=24,
                )
                
                prediction_times = LSTM_forecast_mean_scaled.time_index
                prediction_values = scaler_glucose.inverse_transform(
                    LSTM_forecast_mean_scaled.values().reshape(-1, 1)
                ).flatten()
                
                # Use XGBoost for confidence intervals since LSTM doesn't provide them
                XGB_forecast_conf = xgb_model.predict(
                    series=target_series,
                    past_covariates=covariates,
                    n=24,
                    num_samples=200
                )
                confidence_lower = XGB_forecast_conf.quantile_timeseries(0.1).values().flatten()
                confidence_upper = XGB_forecast_conf.quantile_timeseries(0.9).values().flatten()
                
                return add_common_elements(fig, prediction_times, prediction_values, 
                                        confidence_lower, confidence_upper)
            
            else:  # Ensemble
                # XGBoost prediction
                XGB_forecast_mean = xgb_model.predict(
                    series=target_series,
                    past_covariates=covariates,
                    n=24
                )
                
                # LSTM prediction
                LSTM_forecast_mean_scaled = lstm_model.predict(
                    series=target_lstm[39:],
                    past_covariates=covs_lstm,
                    n=24,
                )
                LSTM_forecast_mean = scaler_glucose.inverse_transform(
                    LSTM_forecast_mean_scaled.values().reshape(-1, 1)
                ).flatten()
                
                # Create ensemble prediction
                prediction_times = XGB_forecast_mean.time_index
                xgb_values = XGB_forecast_mean.values().flatten()
                lstm_values = LSTM_forecast_mean
                
                # Ensure arrays are the same length
                min_len = min(len(xgb_values), len(lstm_values))
                xgb_values = xgb_values
                lstm_values = lstm_values
                
                # Apply optimal weights
                weights = np.array([optimal_weights.get(i+1, 0.5) for i in range(min_len)])
                prediction_values = weights * xgb_values + (1 - weights) * lstm_values
                
                # Confidence intervals from XGBoost
                XGB_forecast_conf = xgb_model.predict(
                    series=target_series,
                    past_covariates=covariates,
                    n=24,
                    num_samples=200
                )
                confidence_lower = XGB_forecast_conf.quantile_timeseries(0.1).values().flatten()[:min_len]
                confidence_upper = XGB_forecast_conf.quantile_timeseries(0.9).values().flatten()[:min_len]
                
                return add_common_elements(fig, prediction_times, prediction_values, 
                                        confidence_lower, confidence_upper)
        
        

        fig = generate_prediction_figure(
            model_choice, xgb_model, lstm_model, target_series, covariates, 
            target_lstm, covs_lstm, scaler_glucose, OPTIMAL_WEIGHTS
        )
        
    except Exception as e:
        print(f"Error generating future prediction: {e}")
        import traceback
        traceback.print_exc()
        return {}, f"Error: {str(e)}"
    

    return fig, "Future prediction generated successfully!"

    
if __name__ == '__main__':
    app.run(debug=True)
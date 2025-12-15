import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots # Necessary for combined plots (Low Fare)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.seasonal import seasonal_decompose

# Page Configurationdame 
st.set_page_config(page_title="SFO Passenger Dashboard", layout="wide")

st.title("**SFO Airport: Terminal Prediction and Analysis**")
st.markdown("Air traffic analysis dashboard using LSTM and structural analysis of terminals.")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    if df['Passenger Count'].dtype == 'O':
        df['Passenger Count'] = df['Passenger Count'].str.replace(',', '').astype(int)
    
    # Basic Preprocessing
    monthly_data = df.groupby('Activity Period')['Passenger Count'].sum().reset_index()
    monthly_data['Date'] = pd.to_datetime(monthly_data['Activity Period'].astype(str), format='%Y%m')
    monthly_data = monthly_data.sort_values('Date')
    
    # Data for terminals
    df["Year"] = df["Activity Period"].astype(str).str[:4].astype(int)
    return df, monthly_data

# Attempt to load data
try:
    df, monthly_data = load_data("Air_Traffic_Passenger_Statistics_20251204.csv")
except FileNotFoundError:
    st.error("CSV file not found. Please ensure 'Air_Traffic_Passenger_Statistics_20251204.csv' is in the same directory.")
    st.stop()

# --- 2. TERMINAL PROCESSING ---
@st.cache_data
def process_terminals(df):
    tabla_terminales = df.pivot_table(
        index="Year", columns="Terminal", values="Passenger Count", aggfunc="sum", fill_value=0
    )
    terminales = ['Terminal 1', 'Terminal 2', 'Terminal 3', 'Other', 'International']
    # Filter only existing columns
    cols_existentes = [c for c in terminales if c in tabla_terminales.columns]
    tabla_terminales = tabla_terminales[cols_existentes]
    
    # Calculate totals and percentages
    tabla_terminales["Dataset Count"] = df.pivot_table(index="Year", values="Passenger Count", aggfunc="sum")
    
    # Data integrity validation
    tabla_terminales["Sum of terminals"] = tabla_terminales[cols_existentes].sum(axis=1)
    tabla_terminales["boleano"] = tabla_terminales["Sum of terminals"] == tabla_terminales["Dataset Count"]
    
    # If everything is correct (True), drop validation columns to clean up the view
    if tabla_terminales["boleano"].all():
        tabla_terminales = tabla_terminales.drop(columns=["Sum of terminals", "boleano"])
    
    # Calculate percentages
    for col in cols_existentes:
        if col == "International": 
            name = "International %"
        else:
            name = f"{col} %"
        tabla_terminales[name] = (tabla_terminales[col] / tabla_terminales["Dataset Count"]) * 100
        
    return tabla_terminales

tabla_terminales = process_terminals(df)

# --- 3. LSTM TRAINING (Cached) ---
@st.cache_resource
def train_lstm_model(data_values, look_back=3, epochs=150): 
    # Data Preparation
    train_size = int(len(data_values) * 0.8)
    train, test = data_values[:train_size], data_values[train_size:]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    def create_dataset(dataset, lookback=1):
        X, Y = [], []
        for i in range(len(dataset) - lookback - 1):
            X.append(dataset[i:i+lookback, 0])
            Y.append(dataset[i + lookback, 0])
        return np.array(X), np.array(Y)

    trainX, trainY = create_dataset(train_scaled, look_back)
    testX, testY = create_dataset(test_scaled, look_back)
    
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
    testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
    
    # Model
    tf.random.set_seed(7)
    model = Sequential()
    model.add(LSTM(32, input_shape=(look_back, 1)))
    model.add(Dropout(0.05))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=0, 
                        validation_data=(testX, testY), callbacks=[early_stop])
    
    # Return 9 values
    return model, scaler, history, trainX, testX, trainY, testY, test_scaled, train_size

# Prepare data for LSTM
dataset_values = monthly_data['Passenger Count'].values.reshape(-1, 1)
look_back = 3

# -------------------
# Congestion Data (Global Prep)
# -------------------
# Filter latest year
latest_year = df['Year'].max()
df_recent_year = df[df['Year'] == latest_year]

# Group data by Terminal and Boarding Area
terminal_bording_area_separated = df_recent_year.groupby(['Terminal', 'Boarding Area'])['Passenger Count'].sum().reset_index()

# Pivot data for matrix format (Heatmap)
heatmap_data_ter_bor = terminal_bording_area_separated.pivot(index='Terminal', columns='Boarding Area', values='Passenger Count')
heatmap_data_ter_bor = heatmap_data_ter_bor.fillna(0) # Fill with 0 where there are no gates


# --- INTERFACE: TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Terminal Trends", "Forecast & Strategy", "Ops & Commercial Insights", "Dependency Analysis", "Risk Assessment"])

# -----------------------------------------------------------------------------
# TAB 1: HISTORICAL ANALYSIS
# -----------------------------------------------------------------------------
with tab1:

    st.subheader("**Seasonal Decomposition Analysis**")
    st.write("Decomposing passenger traffic into Trend (long-term direction), Seasonality (repeating patterns), and Residuals (noise).")

    # Preparar datos para descomposición (requiere índice de fecha)
    decomp_data = monthly_data.set_index('Date')
    
    # Realizar descomposición (Modelo aditivo o multiplicativo)
    # Usamos 'additive' si la amplitud de la estacionalidad no cambia mucho con la tendencia
    decomposition = seasonal_decompose(decomp_data['Passenger Count'], model='additive', period=12)

    # Crear gráfico compuesto con Plotly
    fig_decomp = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                               subplot_titles=("Trend", "Seasonality", "Residuals"))

    # 1. Trend
    fig_decomp.add_trace(
        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend', line=dict(color='blue')),
        row=1, col=1
    )

    # 2. Seasonality
    fig_decomp.add_trace(
        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonality', line=dict(color='green')),
        row=2, col=1
    )

    # 3. Residuals
    fig_decomp.add_trace(
        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='markers', name='Residuals', marker=dict(color='gray', size=4)),
        row=3, col=1
    )

    fig_decomp.update_layout(height=600, title_text="Time Series Decomposition", showlegend=False, template="plotly_white")
    st.plotly_chart(fig_decomp, use_container_width=True)
    
    st.info("💡 **Insight:** The 'Trend' component shows the underlying growth regardless of the time of year, while 'Seasonality' highlights the predictable peaks (e.g., Summer/Holidays).")

    st.subheader("**Terminal Distribution Analysis**")

    col1, col2 = st.columns(2)
    
    # Chart 1: Percentage Evolution
    cols_porcentaje = [c for c in tabla_terminales.columns if "%" in c]
    fig_pct = go.Figure()
    for col in cols_porcentaje:
        fig_pct.add_trace(go.Scatter(x=tabla_terminales.index, y=tabla_terminales[col], mode="lines+markers", name=col))
    fig_pct.update_layout(title="Market Share Evolution by Terminal (%)", template="plotly_white", height=400)
    col1.plotly_chart(fig_pct, use_container_width=True)
    
    # Chart 2: Standard Deviation
    std_dev_por_anio = tabla_terminales[cols_porcentaje].std(axis=1)
    fig_std = px.line(x=std_dev_por_anio.index, y=std_dev_por_anio.values, 
                      title="Imbalance Index (Std Dev)", labels={"x": "Year", "y": "Deviation"})
    fig_std.update_traces(mode="lines+markers", line_color="#8F9C00")
    fig_std.update_layout(template="plotly_white", height=400)
    col2.plotly_chart(fig_std, use_container_width=True)

    # Dataset
    st.write("##### **Terminal % Dataset Used**")
    st.write("Detailed processed data on traffic distribution by terminals and years.")
    
    # Fill null values with 0 for cleaner visualization
    tabla_terminales_view = tabla_terminales.fillna(0)

    # Display dataframe interactively
    st.dataframe(tabla_terminales_view, use_container_width=True)

    # Button to download processed CSV
    csv = tabla_terminales_view.to_csv().encode('utf-8')
    st.download_button(
        label="Download Processed CSV",
        data=csv,
        file_name='terminal_analysis_sfo.csv',
        mime='text/csv',
    )

# -----------------------------------------------------------------------------
# TAB 2: PREDICTIONS (LSTM MODEL)
# -----------------------------------------------------------------------------
with tab2:
    st.header("Neural Model Training")
    
    col_params1, col_params2 = st.columns(2)
    with col_params1:
        epochs_input = st.slider("Epochs", 50, 500, 150)
    with col_params2:
        neurons = st.slider("Neurons", 2, 128, 32)

    if st.button("Train Model") or 'model' not in st.session_state:
        with st.spinner('Training LSTM... this may take a moment'):
            model, scaler, history, trainX, testX, trainY, testY, test_scaled, train_size = train_lstm_model(dataset_values, look_back, epochs=epochs_input)
            
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['history'] = history
            st.session_state['trainX'] = trainX
            st.session_state['testX'] = testX
            st.session_state['trainY'] = trainY
            st.session_state['testY'] = testY
            st.session_state['test_scaled'] = test_scaled
            st.session_state['train_size'] = train_size 
            
            st.success("Model trained successfully")

    if 'model' in st.session_state:
        # Retrieve state data
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        trainX = st.session_state['trainX']
        testX = st.session_state['testX']
        trainY = st.session_state['trainY']
        testY = st.session_state['testY']
        train_size = st.session_state['train_size']
        hist = st.session_state['history']

        # --- SECTION 1: Loss Curve ---
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=hist.history['loss'], name='Train Loss', line=dict(color='#FF3344')))
        fig_loss.add_trace(go.Scatter(y=hist.history['val_loss'], name='Validation Loss', line=dict(color='#00CC96')))
        fig_loss.update_layout(title="Learning Curve (Loss)", height=400, template="plotly_white")
        st.plotly_chart(fig_loss, use_container_width=True)

        # --- SECTION 2: Validation (RMSE and Comparative Chart) ---
        st.subheader("Model Validation")
        
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        trainY_inv = scaler.inverse_transform(trainY.reshape(-1,1))
        testY_inv  = scaler.inverse_transform(testY.reshape(-1,1))
        trainPredict_inv = scaler.inverse_transform(trainPredict)
        testPredict_inv  = scaler.inverse_transform(testPredict)

        train_rmse = np.sqrt(mean_squared_error(trainY_inv, trainPredict_inv))
        test_rmse  = np.sqrt(mean_squared_error(testY_inv,  testPredict_inv))

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Train RMSE (Training Error)", f"{train_rmse:,.0f}")
        col_m2.metric("Test RMSE (Validation Error)", f"{test_rmse:,.0f}")

        # --- Actual vs Predicted Chart ---
        trainPredictPlot = np.empty_like(dataset_values, dtype=float)
        trainPredictPlot[:] = np.nan
        trainPredictPlot[look_back:len(trainPredict_inv)+look_back, 0] = trainPredict_inv[:,0]

        testPredictPlot = np.empty_like(dataset_values, dtype=float)
        testPredictPlot[:] = np.nan
        test_start = train_size + look_back
        testPredictPlot[test_start:test_start+len(testPredict_inv), 0] = testPredict_inv[:,0]

        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(y=dataset_values[:,0], mode='lines', name='Actual Data', line=dict(color="#32C7F9")))
        fig_val.add_trace(go.Scatter(y=trainPredictPlot[:,0], mode='lines', name='Train Prediction', line=dict(color="#FD4A59", width=3)))
        fig_val.add_trace(go.Scatter(y=testPredictPlot[:,0], mode='lines', name='Test Prediction', line=dict(color='#00CC96', width=3)))

        fig_val.update_layout(title='Comparison: Actual Data vs LSTM Model', xaxis_title='Time (Months)', yaxis_title='Passengers', template='plotly_white', legend=dict(x=0, y=1, orientation="h"), height=500)
        st.plotly_chart(fig_val, use_container_width=True)

        st.header("Prediction & Strategy")
    
        months = st.slider("Months to forecast", 1, 24, 14)

        # --- Future Prediction Logic ---
        future_months = months
        future_predictions = []
        
        full_scaled = scaler.transform(dataset_values)
        last_window = full_scaled[-look_back:]
        current_batch = last_window.reshape((1, look_back, 1))
        
        for i in range(future_months):
            current_pred = model.predict(current_batch, verbose=0)[0]
            future_predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
            
        future_predictions_inv = scaler.inverse_transform(future_predictions)
        
        last_date = monthly_data['Date'].iloc[-1]
        future_dates = [last_date + DateOffset(months=x) for x in range(1, future_months + 1)]
        
        # LSTM Projection Chart
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=monthly_data['Date'], y=dataset_values.flatten(), name='Historical', line=dict(color='#32C7F9')))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_predictions_inv.flatten(), name='Projection', 
                                          mode='lines+markers', line=dict(color="#70DE17", width=3, dash='dash')))
        fig_forecast.update_layout(title=f"LSTM Projection ({future_months} months)", template="plotly_white")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # --- Combined Logic (LSTM + Terminals) ---
        st.subheader("Impact on Terminals")
        
        months_to_sum = min(12, future_months)
        pred_anual = np.sum(future_predictions_inv[:months_to_sum])
        
        cols_porcentaje = [c for c in tabla_terminales.columns if "%" in c]
        dist_ultimo_anio = tabla_terminales.iloc[-1][cols_porcentaje]
        
        pasajeros_futuros = (dist_ultimo_anio / 100) * pred_anual
        desequilibrio_futuro = dist_ultimo_anio.std()
        
        col_metrics1, col_metrics2 = st.columns(2)
        col_metrics1.metric(f"Predicted Total ({months_to_sum} months)", f"{pred_anual:,.0f}")
        col_metrics2.metric("Expected Imbalance Index", f"{desequilibrio_futuro:.2f}")
        
        # Plotly Chart
        std_dev_hist = tabla_terminales[cols_porcentaje].std(axis=1)
        ultimo_anio = int(tabla_terminales.index[-1])
        anio_futuro = ultimo_anio + 1
        
        fig_comb = go.Figure()
        fig_comb.add_trace(go.Scatter(x=tabla_terminales.index.astype(int), y=std_dev_hist, mode='lines+markers', name='Historical', line=dict(color='#1f77b4')))
        fig_comb.add_trace(go.Scatter(x=[ultimo_anio, anio_futuro], y=[std_dev_hist.iloc[-1], desequilibrio_futuro], mode='lines+markers', name=f'Projection {anio_futuro}', line=dict(color='red', dash='dash'), marker=dict(symbol='star', size=12, color='red')))
         
        fig_comb.update_layout(title=f"Operational Balance Projection: Year {anio_futuro}", xaxis_title="Year", yaxis_title="Imbalance Index", template='plotly_white', legend=dict(x=0, y=1), height=500)
        st.plotly_chart(fig_comb, use_container_width=True)

        st.markdown("### Predictive Scenario per Terminal")
        st.write(pasajeros_futuros.apply(lambda x: f"{x:,.0f} passengers"))
        
    else:
        st.warning("Please train the model in the 'Neural Model Training' tab first.")

# -----------------------------------------------------------------------------
# TAB 3: CONGESTION & COMMERCIAL
# -----------------------------------------------------------------------------
with tab3:
    st.header("Congestion & Commercial Analysis")
    
    # --- 1. PHYSICAL CONGESTION HEATMAP ---
    st.subheader("Physical Congestion (Gate Level)")
    st.write(f"Passenger volume distribution by boarding area for **{latest_year}**.")

    # heatmap_data_ter_bor dataframe was created globally

    colorscale_gate = [
    [0.0,  "#e8f5e9"],   # verde muy claro
    [0.3,  "#81c784"],   # verde medio
    [0.6,  "#388e3c"],   # verde intenso
    [0.8,  "#ffb3b3"],   # rosa pálido (zona de alerta)
    [1.0,  "#b71c1c"]    # rojo crítico
]
    colorscale_softglacier = [
    [0.0,  "#f4f8fb"],   # azul hielo casi blanco
    [0.25, "#d9e6ef"],   # azul pastel muy suave
    [0.5,  "#b9cfdd"],   # celeste grisáceo
    [0.75, "#8fb3c7"],   # azul glaciar suave
    [1.0,  "#5a7f98"]    # azul acero apagado
]


    fig_heat = px.imshow(
        heatmap_data_ter_bor,
        labels=dict(x="Boarding Area (Gate)", y="Terminal", color="Total Passengers"),
        x=heatmap_data_ter_bor.columns,
        y=heatmap_data_ter_bor.index,
        color_continuous_scale='OrRd', # Red for high traffic
        text_auto='.2s', 
        title=f"Terminal Congestion Heatmap ({latest_year})"
    )
    fig_heat.update_layout(xaxis_title="Boarding Area", yaxis_title="Terminal", height=600, template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Bottleneck Alert
    top_bottlenecks = terminal_bording_area_separated.sort_values(by='Passenger Count', ascending=False).head(3)
    st.markdown("### Top 3 Critical Congestion Zones")
    for index, row in top_bottlenecks.iterrows():
        st.markdown(f"- **{row['Terminal']} | Area {row['Boarding Area']}**: {row['Passenger Count']:,} passengers")

    st.markdown("---")

    # --- 2. AIRLINE HEATMAP ---
    st.subheader("Airline Distribution")
    
    # Create combined column
    df_recent_year["Terminal_Area"] = df_recent_year["Terminal"] + " - " + df_recent_year["Boarding Area"].astype(str)

    pivot_terminal_area = df_recent_year.pivot_table(
        index="Terminal_Area",
        columns="Operating Airline",
        values="Passenger Count",
        aggfunc="sum",
        fill_value=0
    )
    
    # Filter top airlines to avoid giant chart
    top_airlines = pivot_terminal_area.sum().sort_values(ascending=False).head(20).index
    pivot_terminal_area = pivot_terminal_area[top_airlines]

    fig_air = px.imshow(
        pivot_terminal_area,
        labels=dict(x="Airline", y="Terminal", color="Passengers"),
        x=pivot_terminal_area.columns,
        y=pivot_terminal_area.index,
        aspect="auto",
        color_continuous_scale='OrRd',
        title="Heatmap: Congestion by Airline and Terminal (Top 20 Airlines)"
    )
    fig_air.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig_air, use_container_width=True)

    st.markdown("---")

    # --- 3. LOW FARE ANALYSIS (COMMERCIAL) ---
    st.subheader("Commercial Insight: Low Cost vs Other (premium)")

    # Prepare data: Evolution of "Low Fare" vs "Other"
    price_data = df.groupby(['Year', 'Price Category Code'])['Passenger Count'].sum().unstack()

    if 'Low Fare' in price_data.columns:
        price_data['Total'] = price_data.sum(axis=1)
        price_data['Low_Fare_Pct'] = (price_data['Low Fare'] / price_data['Total']) * 100

        # Create Dual Axis Chart
        fig_comm = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar: Total Volume
        fig_comm.add_trace(
            go.Bar(name="Total Traffic", x=price_data.index, y=price_data['Total'], marker_color='lightgrey'),
            secondary_y=False,
        )

        # Line: Low Cost Percentage
        fig_comm.add_trace(
            go.Scatter(name="% Low Cost", x=price_data.index, y=price_data['Low_Fare_Pct'], 
                    mode='lines+markers', marker_color='red', line=dict(width=3)),
            secondary_y=True,
        )

        fig_comm.update_layout(title_text="<b>Passenger Profile Evolution:</b> Low Cost or Other (Premium)?", height=500, template='plotly_white', legend=dict(x=0.01, y=0.99))
        fig_comm.update_yaxes(title_text="Total Passengers", secondary_y=False)
        fig_comm.update_yaxes(title_text="Low Cost Market Share (%)", secondary_y=True)

        st.plotly_chart(fig_comm, use_container_width=True)

        # Automated Insight
        current_share = price_data['Low_Fare_Pct'].iloc[-1]
        peak_share = price_data['Low_Fare_Pct'].max()
        drop = peak_share - current_share

        col_ins1, col_ins2 = st.columns([1, 2])
        with col_ins1:
            st.metric("Current Low Cost Share", f"{current_share:.1f}%", delta=f"-{drop:.1f} pts from peak")
        with col_ins2:
            if current_share < 10:
                st.info("💡 **STRATEGIC RECOMMENDATION:** Passenger profile is becoming 'Premium'. Recommend **increasing Luxury Retail and VIP Services**.") 
            else:
                st.success("**STRATEGIC RECOMMENDATION:** Maintain mixed commercial offering (Standard + Premium).")
    else:
        st.warning("No 'Low Fare' data found in 'Price Category Code' column.")

# -----------------------------------------------------------------------------
# TAB 4: DEPENDENCY ANALYSIS
# -----------------------------------------------------------------------------
with tab4:
    st.header("Dependency Analysis")
    st.markdown("Correlation analysis between terminals to understand traffic dependencies. A high correlation (close to 1) implies traffic moves in sync.")

    # 1. Prepare monthly data for correlation
    monthly_pivot = df.pivot_table(index='Activity Period', columns='Terminal', values='Passenger Count', aggfunc='sum', fill_value=0)
    
    target_terminals = ['Terminal 1', 'Terminal 2', 'Terminal 3', 'International']
    target_terminals = [t for t in target_terminals if t in monthly_pivot.columns]
    monthly_pivot_filtered = monthly_pivot[target_terminals]

    # 2. Calculate correlation matrix
    corr_matrix = monthly_pivot_filtered.corr()

    # 3. Plot Matrix
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='blues',
        title="Correlation Matrix: Passenger Traffic between Terminals",
        aspect="auto"
    )
    fig_corr.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

    # 4. Scatter Matrix
    st.markdown("#### Scatter Matrix")
    st.write("Visual relationship between traffic of major terminals.")
    fig_scatter = px.scatter_matrix(
        monthly_pivot_filtered,
        dimensions=target_terminals,
        title="Scatter Matrix of Terminal Traffic",
        opacity=0.5
    )

    fig_scatter.update_layout(height=700, template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: RISK SIMULATION
# -----------------------------------------------------------------------------
with tab5:
    st.header("Risk Simulation: Airline Dependency")
    st.markdown("Simulation of the impact of a potential **20% reduction** in operations by the dominant airline.")

    latest_year = df['Year'].max()
    df_risk = df[df['Year'] == latest_year]

    # 1. Calculate Market Share
    airline_mkt = df_risk.groupby('Operating Airline')['Passenger Count'].sum().sort_values(ascending=False)
    total_pax = airline_mkt.sum()
    market_share = (airline_mkt / total_pax) * 100

    # Top 5
    top_5 = market_share.head(5)
    dominant_airline = top_5.index[0]
    dominant_share = top_5.iloc[0]

    # 2. Simulation
    drop_scenario_pct = 0.20
    lost_pax = airline_mkt[dominant_airline] * drop_scenario_pct
    projected_total = total_pax - lost_pax

    # Visualization of Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Dominant Airline", f"{dominant_airline}", f"{dominant_share:.1f}% Share")
    c2.metric("Potential Loss (-20%)", f"-{lost_pax/1e6:.2f}M Passengers", delta_color="inverse")
    c3.metric("Projected Total Traffic", f"{projected_total/1e6:.2f}M Passengers")

    st.markdown("---")

    # 3. Waterfall Chart
    fig_waterfall = go.Figure(go.Waterfall(
        name = "Risk Impact",
        orientation = "v",
        measure = ["absolute", "relative", "total"],
        x = ["Current Traffic", f"Drop {dominant_airline} (-20%)", "Projected Traffic"],
        textposition = "outside",
        text = [f"{total_pax/1e6:.2f}M", f"-{lost_pax/1e6:.2f}M", f"{projected_total/1e6:.2f}M"],
        y = [total_pax, -lost_pax, projected_total],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig_waterfall.update_layout(
        title=f"Risk Scenario: Impact of 20% drop in {dominant_airline} operations ({latest_year})",
        showlegend = False,
        height=500,
        template="plotly_white",
        yaxis_title="Total Passengers"
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    # 4. Automated Interpretation
    st.subheader("Automated Strategic Assessment")
    
    if dominant_share > 40:
        risk_level = "CRITICAL"
        risk_icon = "🔴"
    elif dominant_share > 20:
        risk_level = "HIGH"
        risk_icon = "🟠"
    else:
        risk_level = "MODERATE"
        risk_icon = "🟢"

    st.markdown(f"### {risk_icon} Risk Level: {risk_level}")
    st.markdown(f"The airport has a high dependency on **{dominant_airline}** ({dominant_share:.1f}% market share).")
    st.markdown(f"- A 20% reduction in {dominant_airline} operations would cause a direct loss of **{lost_pax:,.0f} passengers**.")
    st.markdown(f"- This loss is equivalent to losing the entire annual traffic of **{top_5.index[2]}** (the 3rd largest airline in the airport).")
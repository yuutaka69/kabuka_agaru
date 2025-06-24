# streamlit_app.py (全機能を単一ファイルに集約 - グラフ英語表記 & 可読性向上 & Plotly混同行列)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import json
from pathlib import Path
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import plotly.express as px # Plotlyのヒートマップ用
import plotly.graph_objects as go # Plotlyの混同行列のカスタマイズ用

# --- Settings ---
GITHUB_RAW_URL_BASE = "https://raw.githubusercontent.com/yuutaka69/kabuka_agaru/main/"
ALL_PERFORMANCE_JSON_URL = f"{GITHUB_RAW_URL_BASE}models/all_stock_model_performance.json"

# --- Data Loading Functions (Cached) ---

@st.cache_data
def load_all_performance_data_from_github():
    """
    Loads all model performance JSON data from GitHub.
    """
    try:
        response = requests.get(ALL_PERFORMANCE_JSON_URL)
        response.raise_for_status() # Check for HTTP errors
        perf_data = response.json()
        st.success(f"Model performance data '{Path(ALL_PERFORMANCE_JSON_URL).name}' loaded from GitHub.")
        return perf_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Could not download model performance data from GitHub: {e}. URL: {ALL_PERFORMANCE_JSON_URL}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Invalid JSON format in model performance file. URL: {ALL_PERFORMANCE_JSON_URL}")
        return None
    except Exception as e:
        st.error(f"Error: An issue occurred while loading model performance: {e}")
        return None

@st.cache_resource
def load_model_and_features_from_github(stock_code, target_period):
    """
    Loads a specific LightGBM model and its trained feature list from GitHub.
    """
    model_filename = f"lgbm_model_{stock_code}_{target_period}d.joblib"
    model_url = f"{GITHUB_RAW_URL_BASE}models/{model_filename}"

    try:
        response = requests.get(model_url)
        response.raise_for_status() # Check for HTTP errors
        
        from io import BytesIO
        loaded_content = joblib.load(BytesIO(response.content))

        if isinstance(loaded_content, tuple) and len(loaded_content) == 2:
            model, feature_names = loaded_content
            return model, feature_names
        else:
            st.error(f"Error: Invalid format for model file '{model_filename}'. Model and feature list are not saved as a tuple.")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Could not download model '{model_filename}' from GitHub: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error: An issue occurred while loading the model: {e}")
        return None, None

@st.cache_data
def load_stock_data_from_github(stock_code):
    """
    Loads stock data CSV file from GitHub and preprocesses it for LightGBM.
    Keeps all features, not just Open, High, Low, Close, Volume.
    """
    data_filename = f"{stock_code}_modified.csv"
    data_url = f"{GITHUB_RAW_URL_BASE}data/modified/{data_filename}"
    
    try:
        df = pd.read_csv(data_url, index_col='Date', parse_dates=True)
        df.dropna(axis='columns', how='all', inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error: Could not download data '{data_filename}' from GitHub: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error: An issue occurred while loading the data file: {e}")
        return pd.DataFrame()

def get_dynamic_threshold_from_metrics(metrics_data):
    """
    Extracts the dynamic threshold percentage string from the metrics dictionary.
    E.g., 'target_14d_4.7p' -> '4.7p'
    """
    if 'target_column_name' in metrics_data:
        full_target_name = metrics_data['target_column_name']
        parts = full_target_name.split('_')
        # Check if parts[2] exists and ends with 'p'
        if len(parts) > 2 and parts[2].endswith('p'):
            return parts[2]
    return "N/A"

def display_metrics_and_confusion_matrix(metrics, title, is_recent=False):
    """
    Helper function to display metrics and confusion matrix.
    Metrics are shown with progress bars, and confusion matrix uses Plotly.
    """
    if not metrics:
        st.warning(f"{title} data not found.")
        return

    st.markdown(f"**{title} Metrics:**")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write(f"- **Accuracy:** `{metrics.get('accuracy', np.nan):.2f}`")
        st.progress(float(metrics.get('accuracy', 0)), text="Accuracy") # Progress bar
        st.write(f"- **ROC AUC Score:** `{metrics.get('roc_auc_score', np.nan):.2f}`")
        st.progress(float(metrics.get('roc_auc_score', 0)), text="ROC AUC Score") # Progress bar
    with col_m2:
        st.markdown(f"**Class 1 (Uplift) Metrics:**")
        if is_recent:
            st.write(f"- Precision: `{metrics.get('precision_class_1', np.nan):.2f}`")
            st.progress(float(metrics.get('precision_class_1', 0)), text="Precision")
            st.write(f"- Recall: `{metrics.get('recall_class_1', np.nan):.2f}`")
            st.progress(float(metrics.get('recall_class_1', 0)), text="Recall")
            st.write(f"- F1-Score: `{metrics.get('f1_score_class_1', np.nan):.2f}`")
            st.progress(float(metrics.get('f1_score_class_1', 0)), text="F1-Score")
        else:
            class_1_metrics = metrics.get('class_1_metrics', {})
            st.write(f"- Precision: `{class_1_metrics.get('precision', np.nan):.2f}`")
            st.progress(float(class_1_metrics.get('precision', 0)), text="Precision")
            st.write(f"- Recall: `{class_1_metrics.get('recall', np.nan):.2f}`")
            st.progress(float(class_1_metrics.get('recall', 0)), text="Recall")
            st.write(f"- F1-Score: `{class_1_metrics.get('f1-score', np.nan):.2f}`")
            st.progress(float(class_1_metrics.get('f1-score', 0)), text="F1-Score")
            
    st.caption(f"※ These metrics are based on the {title.lower()} data.")

    # Confusion Matrix using Plotly
    cm = metrics.get('confusion_matrix', None)
    if cm:
        st.markdown(f"**{title} Confusion Matrix:**")
        cm_df = pd.DataFrame(cm, 
                             index=['Actual: No Rise (0)', 'Actual: Rise (1)'],
                             columns=['Predicted: No Rise (0)', 'Predicted: Rise (1)'])

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_df.values,
            x=cm_df.columns,
            y=cm_df.index,
            colorscale='Blues',
            text=cm_df.values,
            texttemplate="%{text}",
            textfont={"size":16}
        ))
        
        fig_cm.update_layout(
            title_text=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed', # Ensure 'Actual: No Rise' is at top
            height=350, # Set a fixed height
            margin=dict(l=50, r=50, t=50, b=50) # Adjust margins
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning(f"{title} confusion matrix data not found.")


# --- Streamlit UI Setup ---

st.set_page_config(
    layout="wide",
    page_title="Stock Prediction Dashboard",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Prediction Dashboard")
st.markdown("This app provides stock prediction information using **pre-trained models and evaluation results stored on GitHub.**")

# Load all model performance data initially
all_performance_data = load_all_performance_data_from_github()

if all_performance_data is None:
    st.error("Application cannot proceed. Failed to load model performance data from GitHub.")
    st.stop()

# Get available stock codes
available_stocks = sorted(list(all_performance_data.keys()))

if not available_stocks:
    st.error("No stock information found in the model performance data loaded from GitHub.")
    st.stop()

# Stock selection in the sidebar
st.sidebar.header("Select Stock")
selected_stock = st.sidebar.selectbox(
    "Choose a stock to analyze:",
    options=available_stocks,
    index=0
)

# Create tabs
tab1, tab2 = st.tabs(["📊 Ranking", "📈 Stock Details"])

# --- Tab 1: Ranking Display ---
with tab1:
    st.header("📊 Stock Prediction Model Ranking")
    st.markdown("Here you can review the performance of trained models in a ranked format.")

    ranking_data = []
    for stock_code, periods_data in all_performance_data.items():
        for period_str, model_data in periods_data.items():
            training_eval = model_data.get('training_evaluation', {})
            recent_eval = model_data.get('recent_data_evaluation', {})
            
            # Get dynamic threshold (prioritize from training_eval)
            dynamic_threshold_str = get_dynamic_threshold_from_metrics(training_eval)
            if dynamic_threshold_str == "N/A": 
                dynamic_threshold_str = get_dynamic_threshold_from_metrics(recent_eval)

            ranking_data.append({
                'Stock Code': stock_code,
                'Prediction Period (Days)': int(period_str.replace('d', '')),
                'Uplift Threshold (%)': dynamic_threshold_str.replace('p', ''), 
                'TrainAcc': training_eval.get('accuracy', np.nan),
                'TrainAUC': training_eval.get('roc_auc_score', np.nan),
                'TrainF1(1)': training_eval.get('class_1_metrics', {}).get('f1-score', np.nan),
                'RecentAcc': recent_eval.get('accuracy', np.nan),
                'RecentAUC': recent_eval.get('roc_auc_score', np.nan),
                'RecentF1(1)': recent_eval.get('f1_score_class_1', np.nan),
                'LastPredDate': recent_eval.get('most_recent_prediction_date', 'N/A'),
                'LastPredValue': recent_eval.get('most_recent_prediction_value', 'N/A'),
                'LastPredProb': recent_eval.get('most_recent_prediction_proba', np.nan)
            })

    if ranking_data:
        df_ranking = pd.DataFrame(ranking_data)

        # Sort options
        sort_by_options = {
            "RecentF1(1) (Desc)": "RecentF1(1)",
            "TrainF1(1) (Desc)": "TrainF1(1)",
            "RecentAcc (Desc)": "RecentAcc",
            "TrainAcc (Desc)": "TrainAcc",
            "Prediction Period (Days) (Asc)": "Prediction Period (Days)",
            "Stock Code (Asc)": "Stock Code"
        }
        
        col1_rank, col2_rank = st.columns([1, 2])
        with col1_rank:
            sort_key_display = st.selectbox("Sort by:", list(sort_by_options.keys()))
        
        sort_column = sort_by_options[sort_key_display]
        ascending = False if "Desc" in sort_key_display else True

        # Ensure numeric columns are actually numeric for sorting
        numeric_cols = ['TrainAcc', 'TrainAUC', 'TrainF1(1)', 'RecentAcc', 'RecentAUC', 'RecentF1(1)', 'LastPredProb']
        for col in numeric_cols:
            if col in df_ranking.columns:
                df_ranking[col] = pd.to_numeric(df_ranking[col], errors='coerce')

        # Perform sorting
        df_ranking_sorted = df_ranking.sort_values(by=sort_column, ascending=ascending, na_position='last')

        st.markdown("---")
        st.subheader("Overall Ranking")
        st.markdown("_(Higher values generally indicate better performance.)_")

        st.dataframe(df_ranking_sorted.style.format({
            'TrainAcc': "{:.2f}", 'TrainAUC': "{:.2f}", 'TrainF1(1)': "{:.2f}",
            'RecentAcc': "{:.2f}", 'RecentAUC': "{:.2f}", 'RecentF1(1)': "{:.2f}",
            'LastPredProb': "{:.2%}"
        }), use_container_width=True)

        st.caption("※ **Uplift Threshold (%)**: If the stock price rises by this percentage or more within N days, it's classified as 'Rise (1)'.")
        st.caption("※ **LastPredValue** of 1 indicates an 'Uplift' signal, 0 indicates 'No Uplift'.")
        st.caption("※ If **RecentF1(1)** is NaN, it means there were no 'Rise (1)' samples in the recent data, or insufficient data for evaluation.")

    else:
        st.warning("Could not generate ranking data. Please check the data on GitHub.")

# --- Tab 2: Stock Details ---
with tab2:
    st.header(f"📈 Stock Details: {selected_stock}")
    st.markdown("View performance, latest predictions, and price trends for the selected stock's models across various prediction periods.")

    df_selected_stock = load_stock_data_from_github(selected_stock)

    if df_selected_stock.empty:
        st.error(f"Could not load stock data for {selected_stock}.")
        st.stop()

    # Recent stock price trend chart only
    st.subheader("Recent Price Trend")
    try:
        mc = mpf.make_marketcolors(up='green', down='red', wick='inherit', edge='inherit', volume='in', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
        
        chart_data_length = min(len(df_selected_stock), 120) 
        fig, axes = mpf.plot(
            df_selected_stock.tail(chart_data_length), 
            type='candle',
            style=s,
            volume=True,
            figscale=1.5,
            returnfig=True
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Error drawing stock chart: {e}")
        st.info("This might be due to insufficient data or an issue with the data format.")


    st.subheader("Model Performance & Latest Predictions per Period")

    periods_data = all_performance_data.get(selected_stock, {})
    if not periods_data:
        st.warning(f"No model performance data found for stock {selected_stock}.")
    else:
        sorted_periods = sorted([int(p.replace('d', '')) for p in periods_data.keys()])

        for target_period in sorted_periods:
            period_str = f"{target_period}d"
            model_data = periods_data[period_str]

            st.markdown(f"---") 
            st.markdown(f"### {target_period}-Day Prediction Model") 

            # Get dynamic threshold
            dynamic_threshold_str = get_dynamic_threshold_from_metrics(model_data.get('training_evaluation', {}))
            if dynamic_threshold_str == "N/A": 
                dynamic_threshold_str = get_dynamic_threshold_from_metrics(model_data.get('recent_data_evaluation', {}))
            
            st.markdown(f"This model predicts whether the stock price will rise by **{dynamic_threshold_str} or more** within {target_period} days.")

            # --- Training Evaluation Section ---
            st.markdown("#### Training Evaluation (Test Set)")
            display_metrics_and_confusion_matrix(model_data.get('training_evaluation', {}), 'Training Evaluation', is_recent=False)

            # --- Recent Data Evaluation & Latest Prediction Section ---
            st.markdown("#### Recent Data Evaluation & Latest Prediction")
            recent_metrics = model_data.get('recent_data_evaluation', {})
            
            if recent_metrics:
                # Metrics with progress bars
                col_rc_metrics1, col_rc_metrics2 = st.columns(2)
                with col_rc_metrics1:
                    st.write(f"- **Evaluation Period (Days):** `{recent_metrics.get('total_evaluated_days', 'N/A')}`")
                    st.write(f"- **Accuracy:** `{recent_metrics.get('accuracy', np.nan):.2f}`")
                    st.progress(float(recent_metrics.get('accuracy', 0)), text="Accuracy")
                    st.write(f"- **ROC AUC Score:** `{recent_metrics.get('roc_auc_score', np.nan):.2f}`")
                    st.progress(float(recent_metrics.get('roc_auc_score', 0)), text="ROC AUC Score")
                with col_rc_metrics2:
                    st.markdown(f"**Class 1 (Uplift) Metrics:**")
                    st.write(f"- Precision: `{recent_metrics.get('precision_class_1', np.nan):.2f}`")
                    st.progress(float(recent_metrics.get('precision_class_1', 0)), text="Precision")
                    st.write(f"- Recall: `{recent_metrics.get('recall_class_1', np.nan):.2f}`")
                    st.progress(float(recent_metrics.get('recall_class_1', 0)), text="Recall")
                    st.write(f"- F1-Score: `{recent_metrics.get('f1_score_class_1', np.nan):.2f}`")
                    st.progress(float(recent_metrics.get('f1_score_class_1', 0)), text="F1-Score")
                
                st.markdown(f"**Latest Prediction (based on {recent_metrics.get('most_recent_prediction_date', 'N/A')} data):**")
                prediction_value = recent_metrics.get('most_recent_prediction_value', 'N/A')
                prediction_proba = recent_metrics.get('most_recent_prediction_proba', np.nan)
                
                if prediction_value == 1:
                    st.success(f"**📈 Uplift Signal!** (Stock price likely to rise by {dynamic_threshold_str} or more within {target_period} days)")
                    st.progress(prediction_proba, text=f"Prediction Probability: {prediction_proba:.2%}")
                elif prediction_value == 0:
                    st.info(f"**📉 No Uplift Signal** (Stock price less likely to rise by {dynamic_threshold_str} or more within {target_period} days)")
                    st.progress(prediction_proba, text=f"Prediction Probability: {prediction_proba:.2%}")
                else:
                    st.warning("Latest prediction value is not available.")

                st.caption("※ These metrics are based on the most recent period data evaluated by the model.")

                display_metrics_and_confusion_matrix(recent_metrics, 'Recent Data Evaluation', is_recent=True)
            else:
                st.warning("Recent data evaluation and latest prediction not found.")

st.sidebar.markdown("---")
st.sidebar.info("Data Source: Public GitHub Repository")
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ by Yuu")

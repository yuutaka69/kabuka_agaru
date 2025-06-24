# streamlit_app.py (All features in a single file - Japanese localization & improved visibility)

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
import plotly.express as px
import plotly.graph_objects as go

# --- Japanese Font and Plotting Settings ---
import japanize_matplotlib

# Call japanize_matplotlib to automatically configure Japanese font settings for matplotlib
japanize_matplotlib.japanize()

# Additionally, set specific Japanese fonts for preference, considering fonts available in Streamlit Cloud.
# If these fonts are not available, japanize_matplotlib will attempt to find alternatives.
plt.rcParams['font.family'] = 'sans-serif' # Default font family
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAexGothic', 'Hiragino Sans', 'Meiryo', 'Yu Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # Prevent garbled characters for minus signs

# --- Configuration ---
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
        st.success(f"モデル性能データ '{Path(ALL_PERFORMANCE_JSON_URL).name}' をGitHubからロードしました。")
        return perf_data
    except requests.exceptions.RequestException as e:
        st.error(f"エラー: GitHubからモデル性能データをダウンロードできませんでした: {e}。URL: {ALL_PERFORMANCE_JSON_URL}")
        return None
    except json.JSONDecodeError:
        st.error(f"エラー: モデル性能JSONファイルの形式が不正です。URL: {ALL_PERFORMANCE_JSON_URL}")
        return None
    except Exception as e:
        st.error(f"エラー: モデル性能のロード中に問題が発生しました: {e}")
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
            st.error(f"エラー: モデルファイル '{model_filename}' の形式が不正です。モデルと特徴量リストがセットで保存されていません。")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"エラー: GitHubからモデル '{model_filename}' をダウンロードできませんでした: {e}")
        return None, None
    except Exception as e:
        st.error(f"エラー: モデルのロード中に問題が発生しました: {e}")
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
        st.error(f"エラー: GitHubからデータ '{data_filename}' をダウンロードできませんでした: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"エラー: データファイルのロード中に問題が発生しました: {e}")
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
        st.warning(f"{title}のデータが見つかりません。")
        return

    st.markdown(f"**{title}の指標:**")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        # Check for NaN before passing to progress bar
        accuracy_val = float(metrics.get('accuracy', 0)) if not np.isnan(metrics.get('accuracy', np.nan)) else 0
        st.progress(accuracy_val, text=f"精度: {metrics.get('accuracy', np.nan):.2f}")
        
        roc_auc_val = float(metrics.get('roc_auc_score', 0)) if not np.isnan(metrics.get('roc_auc_score', np.nan)) else 0
        st.progress(roc_auc_val, text=f"ROC AUC: {metrics.get('roc_auc_score', np.nan):.2f}")
    with col_m2:
        if is_recent: # Recent data has flat precision_class_1, recall_class_1
            precision_val = float(metrics.get('precision_class_1', 0)) if not np.isnan(metrics.get('precision_class_1', np.nan)) else 0
            st.progress(precision_val, text=f"適合率(クラス1): {metrics.get('precision_class_1', np.nan):.2f}")
            
            recall_val = float(metrics.get('recall_class_1', 0)) if not np.isnan(metrics.get('recall_class_1', np.nan)) else 0
            st.progress(recall_val, text=f"再現率(クラス1): {metrics.get('recall_class_1', np.nan):.2f}")
            
            f1_val = float(metrics.get('f1_score_class_1', 0)) if not np.isnan(metrics.get('f1_score_class_1', np.nan)) else 0
            st.progress(f1_val, text=f"F1スコア(クラス1): {metrics.get('f1_score_class_1', np.nan):.2f}")
        else: # Training data has class_1_metrics nested dict
            class_1_metrics = metrics.get('class_1_metrics', {})
            
            precision_val = float(class_1_metrics.get('precision', 0)) if not np.isnan(class_1_metrics.get('precision', np.nan)) else 0
            st.progress(precision_val, text=f"適合率(クラス1): {class_1_metrics.get('precision', np.nan):.2f}")
            
            recall_val = float(class_1_metrics.get('recall', 0)) if not np.isnan(class_1_metrics.get('recall', np.nan)) else 0
            st.progress(recall_val, text=f"再現率(クラス1): {class_1_metrics.get('recall', np.nan):.2f}")
            
            f1_val = float(class_1_metrics.get('f1-score', 0)) if not np.isnan(class_1_metrics.get('f1-score', np.nan)) else 0
            st.progress(f1_val, text=f"F1スコア(クラス1): {class_1_metrics.get('f1-score', np.nan):.2f}")
            
    st.caption(f"※ これらの指標は{title.lower()}データに基づいています。")

    # Confusion Matrix using Plotly
    cm = metrics.get('confusion_matrix', None)
    if cm:
        st.markdown(f"**{title} 混同行列:**")
        cm_df = pd.DataFrame(cm, 
                             index=['実際: 非上昇 (0)', '実際: 上昇 (1)'],
                             columns=['予測: 非上昇 (0)', '予測: 上昇 (1)'])

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
            xaxis_title="予測",
            yaxis_title="実際",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed', # '実際: 非上昇' が上に来るようにする
            height=350, # Set a fixed height
            margin=dict(l=50, r=50, t=50, b=50) # Adjust margins
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning(f"{title}の混同行列データが見つかりません。")

# --- Streamlit UI Setup ---

st.set_page_config(
    layout="wide",
    page_title="株価予測ダッシュボード",
    initial_sidebar_state="expanded"
)

st.title("📈 株価予測ダッシュボード")
st.markdown("このアプリは、**GitHubに保存された学習済みモデルと評価結果**を使用して、株価予測情報を提供します。")

# Load all model performance data initially
all_performance_data = load_all_performance_data_from_github()

if all_performance_data is None:
    st.error("アプリケーションを続行できません。GitHubからモデル性能データをロードできませんでした。")
    st.stop()

# Get available stock codes
available_stocks = sorted(list(all_performance_data.keys()))

if not available_stocks:
    st.error("GitHubからロードされたモデル性能データに銘柄情報が見つかりません。")
    st.stop()

# Stock selection in the sidebar
st.sidebar.header("銘柄選択")
selected_stock = st.sidebar.selectbox(
    "分析したい銘柄を選択してください:",
    options=available_stocks,
    index=0
)

# Create tabs
tab1, tab2 = st.tabs(["📊 ランキング", "📈 個別銘柄詳細"])

# --- Tab 1: Ranking Display ---
with tab1:
    st.header("📊 株価予測モデル ランキング")
    st.markdown("ここでは、学習済みモデルのパフォーマンスをランキング形式で確認できます。")

    ranking_data = []
    for stock_code, periods_data in all_performance_data.items():
        for period_str, model_data in periods_data.items():
            training_eval = model_data.get('training_evaluation', {})
            recent_eval = model_data.get('recent_data_evaluation', {})
            
            # Get dynamic threshold (prioritize from training_eval)
            dynamic_threshold_str = get_dynamic_threshold_from_metrics(training_eval)
            if dynamic_threshold_str == "N/A": # If not in training_eval, check recent_eval
                dynamic_threshold_str = get_dynamic_threshold_from_metrics(recent_eval)

            ranking_data.append({
                '銘柄コード': stock_code,
                '予測期間 (日)': int(period_str.replace('d', '')),
                '上昇閾値 (%)': dynamic_threshold_str.replace('p', ''), # 'p' を除去して表示
                '訓練Acc': training_eval.get('accuracy', np.nan),
                '訓練AUC': training_eval.get('roc_auc_score', np.nan),
                '訓練F1(1)': training_eval.get('class_1_metrics', {}).get('f1-score', np.nan),
                '直近Acc': recent_eval.get('accuracy', np.nan),
                '直近AUC': recent_eval.get('roc_auc_score', np.nan),
                '直近F1(1)': recent_eval.get('f1_score_class_1', np.nan),
                '最終予測日': recent_eval.get('most_recent_prediction_date', 'N/A'),
                '最終予測値': recent_eval.get('most_recent_prediction_value', 'N/A'),
                '最終予測確率': recent_eval.get('most_recent_prediction_proba', np.nan)
            })

    if ranking_data:
        df_ranking = pd.DataFrame(ranking_data)

        # Sort options
        sort_by_options = {
            "直近F1(1) (降順)": "直近F1(1)",
            "訓練F1(1) (降順)": "訓練F1(1)",
            "直近Acc (降順)": "直近Acc",
            "訓練Acc (降順)": "訓練Acc",
            "予測期間 (日) (昇順)": "予測期間 (日)",
            "銘柄コード (昇順)": "銘柄コード"
        }
        
        col1_rank, col2_rank = st.columns([1, 2])
        with col1_rank:
            sort_key_display = st.selectbox("ソート基準:", list(sort_by_options.keys()))
        
        sort_column = sort_by_options[sort_key_display]
        ascending = False if "降順" in sort_key_display else True

        # Ensure numeric columns are actually numeric for sorting
        numeric_cols = ['訓練Acc', '訓練AUC', '訓練F1(1)', '直近Acc', '直近AUC', '直近F1(1)', '最終予測確率']
        for col in numeric_cols:
            if col in df_ranking.columns:
                df_ranking[col] = pd.to_numeric(df_ranking[col], errors='coerce')

        # Perform sorting
        df_ranking_sorted = df_ranking.sort_values(by=sort_column, ascending=ascending, na_position='last')

        st.markdown("---")
        st.subheader("全体ランキング")
        st.markdown("（各指標は高ければ高いほど良い傾向を示します。）")

        st.dataframe(df_ranking_sorted.style.format({
            '訓練Acc': "{:.2f}", '訓練AUC': "{:.2f}", '訓練F1(1)': "{:.2f}",
            '直近Acc': "{:.2f}", '直近AUC': "{:.2f}", '直近F1(1)': "{:.2f}",
            '最終予測確率': "{:.2%}"
        }), use_container_width=True)

        st.caption("※ **上昇閾値 (%)**: N日後に株価がこの割合以上上昇した場合に「上昇シグナル（1）」と判定されます。")
        st.caption("※ **最終予測値** 1 は「上昇」シグナル、0 は「非上昇」シグナルを示します。")
        st.caption("※ 「直近F1(1)」が NaN の場合、直近データにクラス1のサンプルがなかったか、評価に必要なデータが不足しています。")

    else:
        st.warning("ランキングデータを生成できませんでした。GitHubのデータを確認してください。")

# --- Tab 2: Stock Details ---
with tab2:
    st.header(f"📈 個別銘柄詳細分析: {selected_stock}")
    st.markdown("選択された銘柄の、各予測期間モデルの性能と最新予測、株価推移を表示します。")

    df_selected_stock = load_stock_data_from_github(selected_stock)

    if df_selected_stock.empty:
        st.error(f"銘柄 {selected_stock} の株価データをロードできませんでした。")
        st.stop()

    # Recent stock price trend chart only
    st.subheader("直近の株価推移")
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
        plt.close(fig) # Free up memory
    except Exception as e:
        st.warning(f"株価チャートの描画中にエラーが発生しました: {e}")
        st.info("データが少なすぎるか、またはデータ形式に問題がある可能性があります。")


    st.subheader("モデル性能と各期間の最新予測")

    periods_data = all_performance_data.get(selected_stock, {})
    if not periods_data:
        st.warning(f"銘柄 {selected_stock} のモデル性能データが見つかりませんでした。")
    else:
        sorted_periods = sorted([int(p.replace('d', '')) for p in periods_data.keys()])

        for target_period in sorted_periods:
            period_str = f"{target_period}d"
            model_data = periods_data[period_str]

            st.markdown(f"---") 
            st.markdown(f"### {target_period}日予測モデル") 

            # Get dynamic threshold
            dynamic_threshold_str = get_dynamic_threshold_from_metrics(model_data.get('training_evaluation', {}))
            if dynamic_threshold_str == "N/A": 
                dynamic_threshold_str = get_dynamic_threshold_from_metrics(model_data.get('recent_data_evaluation', {}))
            
            st.markdown(f"このモデルは、**{target_period}日後に株価が{dynamic_threshold_str}以上上昇するか**を予測します。")

            # --- Training Evaluation Section ---
            st.markdown("#### 訓練時の評価 (テストセット)")
            display_metrics_and_confusion_matrix(model_data.get('training_evaluation', {}), '訓練時評価', is_recent=False)

            # --- Recent Data Evaluation & Latest Prediction Section ---
            st.markdown("#### 直近データ評価と最新予測")
            recent_metrics = model_data.get('recent_data_evaluation', {})
            
            if recent_metrics:
                # 評価期間はテキストで
                st.write(f"**評価期間:** `{recent_metrics.get('total_evaluated_days', 'N/A')}` 日間")
                
                # Metrics with progress bars (values are handled to be within 0-1)
                col_rc_metrics1, col_rc_metrics2 = st.columns(2)
                with col_rc_metrics1:
                    # Explicitly handle NaN for progress bar value, keep original for text
                    accuracy_val = float(recent_metrics.get('accuracy', 0)) if not np.isnan(recent_metrics.get('accuracy', np.nan)) else 0
                    st.progress(accuracy_val, text=f"精度: {recent_metrics.get('accuracy', np.nan):.2f}")
                    
                    roc_auc_val = float(recent_metrics.get('roc_auc_score', 0)) if not np.isnan(recent_metrics.get('roc_auc_score', np.nan)) else 0
                    st.progress(roc_auc_val, text=f"ROC AUC: {recent_metrics.get('roc_auc_score', np.nan):.2f}")
                with col_rc_metrics2:
                    precision_val = float(recent_metrics.get('precision_class_1', 0)) if not np.isnan(recent_metrics.get('precision_class_1', np.nan)) else 0
                    st.progress(precision_val, text=f"適合率(クラス1): {recent_metrics.get('precision_class_1', np.nan):.2f}")
                    
                    recall_val = float(recent_metrics.get('recall_class_1', 0)) if not np.isnan(recent_metrics.get('recall_class_1', np.nan)) else 0
                    st.progress(recall_val, text=f"再現率(クラス1): {recent_metrics.get('recall_class_1', np.nan):.2f}")
                    
                    f1_val = float(recent_metrics.get('f1_score_class_1', 0)) if not np.isnan(recent_metrics.get('f1_score_class_1', np.nan)) else 0
                    st.progress(f1_val, text=f"F1スコア(クラス1): {recent_metrics.get('f1_score_class_1', np.nan):.2f}")
                
                st.markdown(f"**最新の予測 ({recent_metrics.get('most_recent_prediction_date', 'N/A')}のデータに基づく):**")
                prediction_value = recent_metrics.get('most_recent_prediction_value', 'N/A')
                prediction_proba = recent_metrics.get('most_recent_prediction_proba', np.nan)
                
                # Prediction probability progress bar (value is handled to be within 0-1)
                proba_val = float(prediction_proba) if not np.isnan(prediction_proba) else 0
                
                if prediction_value == 1:
                    st.success(f"**📈 上昇シグナル！** ({target_period}日後までに株価が{dynamic_threshold_str}以上上昇する可能性が高い)")
                    st.progress(proba_val, text=f"予測確率: {prediction_proba:.2%}")
                elif prediction_value == 0:
                    st.info(f"**📉 非上昇シグナル** ({target_period}日後までに株価が{dynamic_threshold_str}以上上昇する可能性は低い)")
                    st.progress(proba_val, text=f"予測確率: {prediction_proba:.2%}")
                else:
                    st.warning("最新の予測値がありません。")

                st.caption("※ これらの指標はモデルが評価した直近期間のデータに基づいています。")

                display_metrics_and_confusion_matrix(recent_metrics, '直近データ評価', is_recent=True)
            else:
                st.warning("直近データ評価と最新予測が見つかりませんでした。")

st.sidebar.markdown("---")
st.sidebar.info("データソース: GitHubの公開リポジトリ")
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ by Yuu")

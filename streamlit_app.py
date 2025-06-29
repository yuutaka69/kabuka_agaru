# streamlit_app.py (全機能を単一ファイルに集約 - 日本語表記 & 可視性向上 & 特徴量重要度 & 目次機能)

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

# --- 日本語フォントと描画設定 ---
import japanize_matplotlib

# matplotlibの日本語フォント設定を自動で行う
japanize_matplotlib.japanize()

# 特定の日本語フォントを優先的に使用する設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAexGothic', 'Hiragino Sans', 'Meiryo', 'Yu Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 負の符号の文字化け防止

# --- 設定 ---
GITHUB_RAW_URL_BASE = "https://raw.githubusercontent.com/yuutaka69/kabuka_agaru/main/"
ALL_PERFORMANCE_JSON_URL = f"{GITHUB_RAW_URL_BASE}models/all_stock_model_performance.json"

# --- データロード関数 (キャッシュ付き) ---

@st.cache_data
def load_all_performance_data_from_github():
    """
    GitHubから全てのモデル性能JSONデータをロードします。
    """
    try:
        response = requests.get(ALL_PERFORMANCE_JSON_URL)
        response.raise_for_status() # HTTPエラーを確認
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
    GitHubから特定のLightGBMモデルと訓練時に使用された特徴量リストをロードします。
    """
    model_filename = f"lgbm_model_{stock_code}_{target_period}d.joblib"
    model_url = f"{GITHUB_RAW_URL_BASE}models/{model_filename}"

    try:
        response = requests.get(model_url)
        response.raise_for_status() # HTTPエラーを確認
        
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
    GitHubから指定された銘柄のCSVファイルをロードします。
    OHLCVデータのみをロードし、Streamlitアプリのチャート描画に使用します。
    """
    data_filename = f"{stock_code}_modified.csv" # OHLCVのみに絞られたCSVを想定
    data_url = f"{GITHUB_RAW_URL_BASE}data/modified/{data_filename}"
    
    try:
        df = pd.read_csv(data_url, index_col='Date', parse_dates=True)
        # modified.csvは既にOHLCVに絞られているため、ここでは列の再選択は不要だが、
        # 念のため最低限のOHLCV列があるか確認
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.warning(f"警告: '{data_filename}' に必要なOHLCV列が見つからない可能性があります。")
            # 不足している場合は最低限の列のみで続行、またはエラー
            df = df[required_cols] # 存在しない列はNaNになる
            
        df.dropna(axis='columns', how='all', inplace=True) # 全てNaNの列を削除
        df.replace([np.inf, -np.inf], np.nan, inplace=True) # 無限大値をNaNに
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"エラー: GitHubからデータ '{data_filename}' をダウンロードできませんでした: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"エラー: データファイルのロード中に問題が発生しました: {e}")
        return pd.DataFrame()

def get_dynamic_threshold_from_metrics(metrics_data):
    """
    評価指標辞書から動的閾値のパーセンテージ文字列を抽出します。
    例: 'target_14d_4.7p' -> '4.7p'
    """
    if 'target_column_name' in metrics_data:
        full_target_name = metrics_data['target_column_name']
        parts = full_target_name.split('_')
        # parts[2]が存在し、'p'で終わることを確認
        if len(parts) > 2 and parts[2].endswith('p'):
            return parts[2]
    return "N/A"

def display_metrics_and_confusion_matrix(metrics, title, is_recent=False):
    """
    評価指標と混同行列を表示するヘルパー関数。
    指標はプログレスバーで表示され、混同行列はPlotlyを使用します。
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
        if is_recent: # 直近データの場合
            precision_val = float(metrics.get('precision_class_1', 0)) if not np.isnan(metrics.get('precision_class_1', np.nan)) else 0
            st.progress(precision_val, text=f"適合率(クラス1): {metrics.get('precision_class_1', np.nan):.2f}")
            
            recall_val = float(metrics.get('recall_class_1', 0)) if not np.isnan(metrics.get('recall_class_1', np.nan)) else 0
            st.progress(recall_val, text=f"再現率(クラス1): {metrics.get('recall_class_1', np.nan):.2f}")
            
            f1_val = float(metrics.get('f1_score_class_1', 0)) if not np.isnan(metrics.get('f1_score_class_1', np.nan)) else 0
            st.progress(f1_val, text=f"F1スコア(クラス1): {metrics.get('f1_score_class_1', np.nan):.2f}")
        else: # 訓練データの場合
            class_1_metrics = metrics.get('class_1_metrics', {})
            
            precision_val = float(class_1_metrics.get('precision', 0)) if not np.isnan(class_1_metrics.get('precision', np.nan)) else 0
            st.progress(precision_val, text=f"適合率(クラス1): {class_1_metrics.get('precision', np.nan):.2f}")
            
            recall_val = float(class_1_metrics.get('recall', 0)) if not np.isnan(class_1_metrics.get('recall', np.nan)) else 0
            st.progress(recall_val, text=f"再現率(クラス1): {class_1_metrics.get('recall', np.nan):.2f}")
            
            f1_val = float(class_1_metrics.get('f1-score', 0)) if not np.isnan(class_1_metrics.get('f1-score', np.nan)) else 0
            st.progress(f1_val, text=f"F1スコア(クラス1): {class_1_metrics.get('f1-score', np.nan):.2f}")
            
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

# --- Streamlit UI設定 ---

st.set_page_config(
    layout="wide",
    page_title="株価予測ダッシュボード",
    initial_sidebar_state="expanded"
)

st.title("📈 株価予測ダッシュボード")
st.markdown("このアプリは、**GitHubに保存された学習済みモデルと評価結果**を使用して、株価予測情報を提供します。")

# 全てのモデル性能データを初期ロード
all_performance_data = load_all_performance_data_from_github()

if all_performance_data is None:
    st.error("アプリケーションを続行できません。GitHubからモデル性能データをロードできませんでした。")
    st.stop()

# 利用可能な銘柄コードを取得
available_stocks = sorted(list(all_performance_data.keys()))

if not available_stocks:
    st.error("GitHubからロードされたモデル性能データに銘柄情報が見つかりません。")
    st.stop()

# サイドバーに銘柄選択を配置
st.sidebar.header("銘柄選択")
selected_stock = st.sidebar.selectbox(
    "分析したい銘柄を選択してください:",
    options=available_stocks,
    index=0
)

# タブの作成
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
            
            # 動的閾値を取得 (training_eval から優先的に取得)
            dynamic_threshold_str = get_dynamic_threshold_from_metrics(training_eval)
            if dynamic_threshold_str == "N/A": # training_evalになければ recent_eval を確認
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

        # ソートオプション
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

        # 数値カラムを適切にソートするために、NaNのままにしておく
        numeric_cols = ['訓練Acc', '訓練AUC', '訓練F1(1)', '直近Acc', '直近AUC', '直近F1(1)', '最終予測確率']
        for col in numeric_cols:
            if col in df_ranking.columns:
                df_ranking[col] = pd.to_numeric(df_ranking[col], errors='coerce')

        # ソート実行
        df_ranking_sorted = df_ranking.sort_values(by=sort_column, ascending=ascending, na_position='last')

        st.markdown("---")
        st.subheader("全体ランキング")
        st.markdown("各指標は高いほどパフォーマンスが良いことを示します。")

        st.dataframe(df_ranking_sorted.style.format({
            '訓練Acc': "{:.2f}", '訓練AUC': "{:.2f}", '訓練F1(1)': "{:.2f}",
            '直近Acc': "{:.2f}", '直近AUC': "{:.2f}", '直近F1(1)': "{:.2f}",
            '最終予測確率': "{:.2%}"
        }), use_container_width=True)

        st.caption("**上昇閾値 (%)**: N日後に株価がこの割合以上上昇した場合に「上昇シグナル（1）」と判定されます。")
        st.caption("**最終予測値** 1 は「上昇」シグナル、0 は「非上昇」シグナルです。")
        st.caption("「直近F1(1)」が NaN の場合、直近データにクラス1のサンプルがなかったか、評価に必要なデータが不足しています。")

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

    # 直近の株価推移チャートのみ表示
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
        plt.close(fig) # メモリ解放
    except Exception as e:
        st.warning(f"株価チャートの描画中にエラーが発生しました: {e}")
        st.info("データが少なすぎるか、またはデータ形式に問題がある可能性があります。")


    st.subheader("モデル性能と各期間の最新予測")

    periods_data = all_performance_data.get(selected_stock, {})
    if not periods_data:
        st.warning(f"銘柄 {selected_stock} のモデル性能データが見つかりませんでした。")
    else:
        sorted_periods = sorted([int(p.replace('d', '')) for p in periods_data.keys()])

        # サイドバーに目次を追加
        st.sidebar.markdown("---")
        st.sidebar.header("期間別モデル概要")
        for period in sorted_periods:
            st.sidebar.markdown(f"[{period}日予測モデル](#{period}日予測モデル)") # アンカーリンク
        st.sidebar.markdown("---")

        for target_period in sorted_periods:
            period_str = f"{target_period}d"
            model_data = periods_data[period_str]

            st.markdown(f"---") 
            # アンカーリンクのターゲットとなるIDを設定
            st.markdown(f"<a name='{target_period}日予測モデル'></a>", unsafe_allow_html=True) 
            st.markdown(f"### {target_period}日予測モデル") 

            # 動的閾値を取得
            dynamic_threshold_str = get_dynamic_threshold_from_metrics(model_data.get('training_evaluation', {}))
            if dynamic_threshold_str == "N/A": 
                dynamic_threshold_str = get_dynamic_threshold_from_metrics(model_data.get('recent_data_evaluation', {}))
            
            st.markdown(f"このモデルは、**{target_period}日後に株価が{dynamic_threshold_str}以上上昇するか**を予測します。")

            # --- 訓練時評価セクション ---
            st.markdown("#### 訓練時評価 (テストセット)")
            display_metrics_and_confusion_matrix(model_data.get('training_evaluation', {}), '訓練時評価', is_recent=False)

            # --- 直近データ評価と最新予測セクション ---
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

                display_metrics_and_confusion_matrix(recent_metrics, '直近データ評価', is_recent=True)
            else:
                st.warning("直近データ評価と最新予測が見つかりませんでした。")
            
            # --- Feature Importance Chart ---
            st.markdown("#### 特徴量重要度")
            # `features_used`はtraining_evaluationに格納されている
            features_used = model_data.get('training_evaluation', {}).get('features_used', None)

            if features_used:
                # モデルファイルをロードして特徴量重要度を取得
                # Note: This might be slow if models are large and not effectively cached.
                # Consider pre-extracting feature importances to JSON if performance is an issue.
                model_obj, _ = load_model_and_features_from_github(selected_stock, target_period)
                
                if model_obj and hasattr(model_obj, 'feature_importances_'):
                    feature_importance = pd.Series(model_obj.feature_importances_, index=features_used)
                    top_features = feature_importance.nlargest(15) # 上位15個の特徴量

                    fig_fi = px.bar(
                        top_features,
                        x=top_features.values,
                        y=top_features.index,
                        orientation='h',
                        title='予測に影響を与えた上位特徴量',
                        labels={'x': '重要度', 'y': '特徴量'},
                        height=400 # チャートの高さ
                    )
                    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}) # 重要度順にソート
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.warning("モデルまたは特徴量重要度データが見つからないため、特徴量重要度チャートを表示できません。")
            else:
                st.warning("モデルの訓練に使用された特徴量リストが見つからないため、特徴量重要度チャートを表示できません。")


st.sidebar.markdown("---")
st.sidebar.info("データソース: GitHubの公開リポジトリ")
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ by Yuu")

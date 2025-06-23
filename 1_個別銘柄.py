# pages/1_個別銘柄.py (個別銘柄詳細ページ)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import json
from pathlib import Path
import matplotlib.pyplot as plt
import mplfinance as mpf # 株価チャート表示用

# --- 設定 ---
GITHUB_RAW_URL_BASE = "https://raw.githubusercontent.com/yuutaka69/kabuka_agaru/main/"
ALL_PERFORMANCE_JSON_URL = f"{GITHUB_RAW_URL_BASE}models/all_stock_model_performance.json"

# --- 関数: データロード ---

@st.cache_data
def load_all_performance_data_from_github():
    """
    GitHubから全体のモデル性能JSONデータをロードする。
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
    GitHubから特定のLightGBMモデルと訓練時に使用された特徴量リストをロードする。
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
            # st.success(f"モデル '{model_filename}' をGitHubからロードしました。")
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
    GitHubから指定された銘柄のCSVファイルをロードし、LightGBMが扱えるように前処理を行う。
    ここでは予測に必要なOpen, High, Low, Close, Volume以外の特徴量も保持する。
    """
    data_filename = f"{stock_code}_modified.csv"
    data_url = f"{GITHUB_RAW_URL_BASE}data/modified/{data_filename}"
    
    try:
        df = pd.read_csv(data_url, index_col='Date', parse_dates=True)
        df.dropna(axis='columns', how='all', inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # st.success(f"データ '{data_filename}' をGitHubからロードしました。")
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"エラー: GitHubからデータ '{data_filename}' をダウンロードできませんでした: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"エラー: データファイルのロード中に問題が発生しました: {e}")
        return pd.DataFrame()

# --- Streamlit UIの構築 ---

st.set_page_config(
    layout="wide",
    page_title="個別銘柄詳細",
    initial_sidebar_state="expanded"
)

st.title("📊 個別銘柄詳細分析")
st.markdown("選択された銘柄の、各予測期間モデルの性能と最新予測を表示します。")

# 全体のモデル性能データを最初にロード
all_performance_data = load_all_performance_data_from_github()

if all_performance_data is None:
    st.error("アプリケーションを続行できません。GitHubからモデル性能データをロードできませんでした。")
    st.stop()

# 利用可能な銘柄リストの取得
available_stocks = sorted(list(all_performance_data.keys()))

if not available_stocks:
    st.error("GitHubからロードされたモデル性能データに銘柄情報が見つかりません。")
    st.stop()

# 銘柄選択（サイドバーに移動）
selected_stock = st.sidebar.selectbox(
    "分析したい銘柄を選択してください:",
    options=available_stocks,
    index=0
)

st.header(f"銘柄: {selected_stock}")

df_selected_stock = load_stock_data_from_github(selected_stock)

if df_selected_stock.empty:
    st.error(f"銘柄 {selected_stock} の株価データをロードできませんでした。")
    st.stop()

# 最新の株価データ表示とチャート
st.subheader("直近の株価推移")
st.dataframe(df_selected_stock[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))

# mplfinanceでチャート描画
try:
    mc = mpf.make_marketcolors(up='red', down='blue', wick='inherit', edge='inherit', volume='in', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
    
    fig, axes = mpf.plot(
        df_selected_stock.tail(120), # 直近120日間のチャートを表示
        type='candle',
        style=s,
        volume=True,
        figscale=1.5,
        returnfig=True
    )
    st.pyplot(fig)
except Exception as e:
    st.warning(f"株価チャートの描画中にエラーが発生しました: {e}")
    st.info("データが少なすぎるか、形式に問題がある可能性があります。")


st.subheader("各予測期間モデルの性能と最新予測")

periods_data = all_performance_data.get(selected_stock, {})
if not periods_data:
    st.warning(f"銘柄 {selected_stock} のモデル性能データが見つかりませんでした。")
else:
    sorted_periods = sorted([int(p.replace('d', '')) for p in periods_data.keys()])

    for target_period in sorted_periods:
        period_str = f"{target_period}d"
        model_data = periods_data[period_str]

        st.markdown(f"---")
        st.markdown(f"#### {target_period}日予測モデル")

        # 訓練時評価
        st.markdown("##### 訓練時の評価 (テストセット)")
        training_metrics = model_data.get('training_evaluation', {})
        if training_metrics:
            st.write(f"- **精度 (Accuracy):** `{training_metrics.get('accuracy', np.nan):.2f}`")
            st.write(f"- **ROC AUC スコア:** `{training_metrics.get('roc_auc_score', np.nan):.2f}`")
            class_1_train = training_metrics.get('class_1_metrics', {})
            st.write(f"- **クラス1 (上昇) F1スコア:** `{class_1_train.get('f1-score', np.nan):.2f}`")
            st.caption("※これらの指標はモデルが訓練・評価された際のテストデータに基づいています。")
        else:
            st.warning("訓練時評価データが見つかりませんでした。")

        # 直近データ評価と最新予測
        st.markdown("##### 直近データ評価と最新予測")
        recent_metrics = model_data.get('recent_data_evaluation', {})
        if recent_metrics:
            st.write(f"- **評価期間 (日数):** `{recent_metrics.get('total_evaluated_days', 'N/A')}`")
            st.write(f"- **精度 (Accuracy):** `{recent_metrics.get('accuracy', np.nan):.2f}`")
            st.write(f"- **ROC AUC スコア:** `{recent_metrics.get('roc_auc_score', np.nan):.2f}`")
            st.write(f"- **クラス1 (上昇) F1スコア:** `{recent_metrics.get('f1_score_class_1', np.nan):.2f}`")
            
            st.markdown(f"**最新の予測 ({recent_metrics.get('most_recent_prediction_date', 'N/A')}のデータに基づく):**")
            prediction_value = recent_metrics.get('most_recent_prediction_value', 'N/A')
            prediction_proba = recent_metrics.get('most_recent_prediction_proba', np.nan)
            
            if prediction_value == 1:
                st.success(f"**📈 上昇シグナル！** ({target_period}日後までに株価が上昇する可能性が高い)")
                st.metric("予測確率", f"{prediction_proba:.2%}" if not np.isnan(prediction_proba) else 'N/A')
            elif prediction_value == 0:
                st.info(f"**📉 非上昇シグナル** ({target_period}日後までに株価が上昇する可能性は低い)")
                st.metric("予測確率", f"{prediction_proba:.2%}" if not np.isnan(prediction_proba) else 'N/A')
            else:
                st.warning("最新の予測値がありません。")

            st.caption("※これらの指標はモデルが評価された直近の期間データに基づいています。")
        else:
            st.warning("直近データ評価と最新予測が見つかりませんでした。")

st.sidebar.markdown("---")
st.sidebar.info("メインページに戻って、他の銘柄のランキングを確認できます。")

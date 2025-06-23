# streamlit_app.py (メインページ - ランキング表示)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path

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

# --- Streamlit UIの構築 ---

st.set_page_config(
    layout="wide",
    page_title="株価予測ダッシュボード - ランキング",
    initial_sidebar_state="expanded"
)

st.title("📈 株価予測モデル ランキング")
st.markdown("ここでは、学習済みモデルのパフォーマンスをランキング形式で確認できます。気になる銘柄をクリックして詳細ページへ。")

# 全体のモデル性能データを最初にロード
all_performance_data = load_all_performance_data_from_github()

if all_performance_data is None:
    st.error("アプリケーションを続行できません。GitHubからモデル性能データをロードできませんでした。")
    st.stop()

# ランキングデータフレームの作成
ranking_data = []
for stock_code, periods_data in all_performance_data.items():
    for period_str, model_data in periods_data.items():
        training_eval = model_data.get('training_evaluation', {})
        recent_eval = model_data.get('recent_data_evaluation', {})

        ranking_data.append({
            '銘柄コード': stock_code,
            '予測期間 (日)': int(period_str.replace('d', '')),
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
    col1, col2 = st.columns([1, 1])
    with col1:
        sort_key_display = st.selectbox("ソート基準:", list(sort_by_options.keys()))
    
    sort_column = sort_by_options[sort_key_display]
    ascending = False if "降順" in sort_key_display else True

    # 数値カラムを適切にソートするために、文字列に変換されていないことを確認
    # そして、NaN値をソート順の最後に来るようにする
    numeric_cols = ['訓練Acc', '訓練AUC', '訓練F1(1)', '直近Acc', '直近AUC', '直近F1(1)', '最終予測確率']
    for col in numeric_cols:
        if col in df_ranking.columns:
            df_ranking[col] = pd.to_numeric(df_ranking[col], errors='coerce') # 強制的に数値に変換、変換できないものはNaNに

    # ソート実行
    df_ranking_sorted = df_ranking.sort_values(by=sort_column, ascending=ascending, na_position='last')

    st.markdown("---")
    st.subheader("全体ランキング")
    st.markdown("（各指標は高ければ高いほど良い傾向を示します。）")

    # DataFrameをStreamlitで表示。style.formatはNaNを自動で無視します。
    st.dataframe(df_ranking_sorted.style.format({
        '訓練Acc': "{:.2f}", '訓練AUC': "{:.2f}", '訓練F1(1)': "{:.2f}",
        '直近Acc': "{:.2f}", '直近AUC': "{:.2f}", '直近F1(1)': "{:.2f}",
        '最終予測確率': "{:.2%}"
    }), use_container_width=True)

    st.caption("※ 最終予測値が1は上昇シグナル、0は非上昇シグナルを示します。")
    st.caption("※ 直近F1(1)が NaN の場合、直近データにクラス1のサンプルがなかったか、評価に必要なデータが不足しています。")

else:
    st.warning("ランキングデータを生成できませんでした。GitHubのデータを確認してください。")

st.sidebar.markdown("---")
st.sidebar.info("左側のメニューから個別の銘柄詳細ページへ移動できます。")
st.sidebar.markdown("Streamlitのマルチページ機能で、`pages/`ディレクトリに各銘柄のファイルを作成すると、ここに自動で表示されます。")

# streamlit_app.py (全機能を単一ファイルに集約)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import json
from pathlib import Path
import matplotlib.pyplot as plt # ここでpltをインポート
import mplfinance as mpf
import seaborn as sns

# --- 日本語フォント設定 ---
# Streamlit Cloud環境ではシステムフォントを直接指定するのが難しい場合があるため、
# Google Fontsなどから提供されるフォント（例: Noto Sans JPなど）をダウンロードして配置するか、
# Matplotlibのデフォルト設定を変更して利用可能なフォントを探す。
# ここでは汎用的なアプローチとして、システムの日本語フォントを検索して設定する例を示します。
# 環境によっては追加のフォントインストール（例: !apt-get install -y fonts-ipafont-gothic）が必要。
import matplotlib.font_manager as fm

@st.cache_resource
def set_japanese_font():
    try:
        # よく使われる日本語フォントのパスをリストアップ (環境依存)
        font_paths = [
            '/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf', # Noto Sans JP (Debian/Ubuntu)
            '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf', # IPAex Gothic (Debian/Ubuntu)
            '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc', # macOS
            '/Library/Fonts/Microsoft Sans Serif.ttf', # Windows
        ]
        
        found_font = None
        for path in font_paths:
            if Path(path).exists():
                found_font = path
                break
        
        if found_font:
            fm.fontManager.addfont(found_font)
            plt.rcParams['font.family'] = fm.FontProperties(fname=found_font).get_name()
            st.sidebar.success(f"日本語フォント '{Path(found_font).name}' を設定しました。")
        else:
            st.sidebar.warning("日本語フォントが見つかりませんでした。グラフの日本語が文字化けする可能性があります。")
            plt.rcParams['font.family'] = 'sans-serif' # デフォルトに戻す
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # fall back to a common font
    except Exception as e:
        st.sidebar.error(f"日本語フォント設定中にエラーが発生しました: {e}")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

set_japanese_font() # アプリ起動時にフォント設定を実行

# --- 設定 ---
GITHUB_RAW_URL_BASE = "https://raw.githubusercontent.com/yuutaka69/kabuka_agaru/main/"
ALL_PERFORMANCE_JSON_URL = f"{GITHUB_RAW_URL_BASE}models/all_stock_model_performance.json"

# --- 関数: データロード (キャッシュ付き) ---

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
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"エラー: GitHubからデータ '{data_filename}' をダウンロードできませんでした: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"エラー: データファイルのロード中に問題が発生しました: {e}")
        return pd.DataFrame()

def get_dynamic_threshold_from_metrics(metrics_data):
    """
    評価指標辞書から動的閾値のパーセンテージ文字列を抽出する。
    例: 'target_14d_4.7p' -> '4.7p'
    """
    if 'target_column_name' in metrics_data:
        full_target_name = metrics_data['target_column_name']
        parts = full_target_name.split('_')
        # parts[2]が存在し、'p'で終わることを確認
        if len(parts) > 2 and parts[2].endswith('p'):
            return parts[2]
    return "N/A"

# --- Streamlit UIの構築 ---

st.set_page_config(
    layout="wide",
    page_title="株価予測ダッシュボード",
    initial_sidebar_state="expanded" # サイドバーを常に展開
)

st.title("📈 株価予測ダッシュボード")
st.markdown("このアプリは、**GitHubに保存された学習済みモデルと評価結果**を使用して、株価予測情報を提供します。")

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

# サイドバーに銘柄選択を配置
st.sidebar.header("銘柄選択")
selected_stock = st.sidebar.selectbox(
    "分析したい銘柄を選択してください:",
    options=available_stocks,
    index=0
)

# タブの作成
tab1, tab2 = st.tabs(["📊 ランキング", "📈 個別銘柄詳細"])

# --- タブ1: ランキング表示 ---
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
                df_ranking[col] = pd.to_numeric(df_ranking[col], errors='coerce') # 強制的に数値に変換、変換できないものはNaNに

        # ソート実行
        df_ranking_sorted = df_ranking.sort_values(by=sort_column, ascending=ascending, na_position='last')

        st.markdown("---")
        st.subheader("全体ランキング")
        st.markdown("（各指標は高ければ高いほど良い傾向を示します。）")

        # DataFrameをStreamlitで表示。style.formatはNaNを自動で無視します。
        # '上昇閾値 (%)' は数値ではないので、数値フォーマットから除外
        st.dataframe(df_ranking_sorted.style.format({
            '訓練Acc': "{:.2f}", '訓練AUC': "{:.2f}", '訓練F1(1)': "{:.2f}",
            '直近Acc': "{:.2f}", '直近AUC': "{:.2f}", '直近F1(1)': "{:.2f}",
            '最終予測確率': "{:.2%}"
        }), use_container_width=True)

        st.caption("※ **上昇閾値 (%)**: N日後に株価がこの割合以上上昇した場合に「上昇シグナル（1）」と判定されます。")
        st.caption("※ 最終予測値が1は上昇シグナル、0は非上昇シグナルを示します。")
        st.caption("※ 直近F1(1)が NaN の場合、直近データにクラス1のサンプルがなかったか、評価に必要なデータが不足しています。")

    else:
        st.warning("ランキングデータを生成できませんでした。GitHubのデータを確認してください。")

# --- タブ2: 個別銘柄詳細 ---
with tab2:
    st.header(f"📈 個別銘柄詳細分析: {selected_stock}")
    st.markdown("選択された銘柄の、各予測期間モデルの性能と最新予測、株価推移を表示します。")

    df_selected_stock = load_stock_data_from_github(selected_stock)

    if df_selected_stock.empty:
        st.error(f"銘柄 {selected_stock} の株価データをロードできませんでした。")
        st.stop()

    # 最新の株価推移チャートのみ表示 (表は削除)
    st.subheader("直近の株価推移")
    try:
        mc = mpf.make_marketcolors(up='red', down='blue', wick='inherit', edge='inherit', volume='in', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
        
        # 描画する期間がデータフレームの長さを超えないように調整
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

            # 動的閾値を取得 (training_eval から優先的に取得)
            dynamic_threshold_str = get_dynamic_threshold_from_metrics(model_data.get('training_evaluation', {}))
            if dynamic_threshold_str == "N/A": # training_evalになければ recent_eval を確認
                dynamic_threshold_str = get_dynamic_threshold_from_metrics(model_data.get('recent_data_evaluation', {}))
            
            st.markdown(f"このモデルは、**{target_period}日後に株価が{dynamic_threshold_str}以上上昇するか**を予測します。")

            # 訓練時評価
            st.markdown("##### 訓練時の評価 (テストセット)")
            
            training_metrics = model_data.get('training_evaluation', {})
            if training_metrics:
                # 指標をカラムで表示
                col_tr_metrics1, col_tr_metrics2 = st.columns(2)
                with col_tr_metrics1:
                    st.write(f"- **精度 (Accuracy):** `{training_metrics.get('accuracy', np.nan):.2f}`")
                    st.write(f"- **ROC AUC スコア:** `{training_metrics.get('roc_auc_score', np.nan):.2f}`")
                with col_tr_metrics2:
                    st.markdown(f"**クラス 1 (上昇する) のメトリクス:**")
                    class_1_train = training_metrics.get('class_1_metrics', {})
                    st.write(f"- 適合率 (Precision): `{class_1_train.get('precision', np.nan):.2f}`")
                    st.write(f"- 再現率 (Recall): `{class_1_train.get('recall', np.nan):.2f}`")
                    st.write(f"- F1スコア: `{class_1_train.get('f1-score', np.nan):.2f}`")
                st.caption("※これらの指標はモデルが訓練・評価された際のテストデータに基づいています。")

                # 混同行列の表示 (訓練時)
                cm_train = training_metrics.get('confusion_matrix', None)
                if cm_train:
                    st.markdown("##### 混同行列 (訓練データ)")
                    fig_cm_train, ax_cm_train = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax_cm_train,
                                xticklabels=['予測:非上昇 (0)', '予測:上昇 (1)'],
                                yticklabels=['実際:非上昇 (0)', '実際:上昇 (1)'])
                    ax_cm_train.set_title('訓練時 Confusion Matrix')
                    ax_cm_train.set_ylabel('実際')
                    ax_cm_train.set_xlabel('予測')
                    st.pyplot(fig_cm_train)
                    plt.close(fig_cm_train) # メモリ解放
                else:
                    st.warning("訓練時混同行列データが見つかりませんでした。")
            else:
                st.warning("訓練時評価データが見つかりませんでした。")

            # 直近データ評価と最新予測
            st.markdown("##### 直近データ評価と最新予測")
            
            recent_metrics = model_data.get('recent_data_evaluation', {})
            if recent_metrics:
                # 指標をカラムで表示
                col_rc_metrics1, col_rc_metrics2 = st.columns(2)
                with col_rc_metrics1:
                    st.write(f"- **評価期間 (日数):** `{recent_metrics.get('total_evaluated_days', 'N/A')}`")
                    st.write(f"- **精度 (Accuracy):** `{recent_metrics.get('accuracy', np.nan):.2f}`")
                    st.write(f"- **ROC AUC スコア:** `{recent_metrics.get('roc_auc_score', np.nan):.2f}`")
                with col_rc_metrics2:
                    st.markdown(f"**クラス 1 (上昇する) のメトリクス:**")
                    st.write(f"- 適合率 (Precision): `{recent_metrics.get('precision_class_1', np.nan):.2f}`")
                    st.write(f"- 再現率 (Recall): `{recent_metrics.get('recall_class_1', np.nan):.2f}`")
                    st.write(f"- F1スコア: `{recent_metrics.get('f1_score_class_1', np.nan):.2f}`")
                
                st.markdown(f"**最新の予測 ({recent_metrics.get('most_recent_prediction_date', 'N/A')}のデータに基づく):**")
                prediction_value = recent_metrics.get('most_recent_prediction_value', 'N/A')
                prediction_proba = recent_metrics.get('most_recent_prediction_proba', np.nan)
                
                if prediction_value == 1:
                    st.success(f"**📈 上昇シグナル！** ({target_period}日後までに株価が{dynamic_threshold_str}以上上昇する可能性が高い)")
                    st.metric("予測確率", f"{prediction_proba:.2%}" if not np.isnan(prediction_proba) else 'N/A')
                elif prediction_value == 0:
                    st.info(f"**📉 非上昇シグナル** ({target_period}日後までに株価が{dynamic_threshold_str}以上上昇する可能性は低い)")
                    st.metric("予測確率", f"{prediction_proba:.2%}" if not np.isnan(prediction_proba) else 'N/A')
                else:
                    st.warning("最新の予測値がありません。")

                st.caption("※これらの指標はモデルが評価された直近の期間データに基づいています。")

                # 混同行列の表示 (直近データ)
                cm_recent = recent_metrics.get('confusion_matrix', None)
                if cm_recent:
                    st.markdown("##### 混同行列 (直近データ)")
                    fig_cm_recent, ax_cm_recent = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm_recent, annot=True, fmt='d', cmap='Blues', ax=ax_cm_recent,
                                xticklabels=['予測:非上昇 (0)', '予測:上昇 (1)'],
                                yticklabels=['実際:非上昇 (0)', '実際:上昇 (1)'])
                    ax_cm_recent.set_title('直近データ Confusion Matrix')
                    ax_cm_recent.set_ylabel('実際')
                    ax_cm_recent.set_xlabel('予測')
                    st.pyplot(fig_cm_recent)
                    plt.close(fig_cm_recent) # メモリ解放
                else:
                    st.warning("直近データ混同行列が見つかりませんでした。")
            else:
                st.warning("直近データ評価と最新予測が見つかりませんでした。")

st.sidebar.markdown("---")
st.sidebar.info("データソース: GitHubの公開リポジトリ")
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ by Yuu") # あなたの名前に変更してください

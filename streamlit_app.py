import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt
import mplfinance as mpf
import json
import requests # GitHubからJSONをダウンロードするために追加

# --- 設定 ---
# ローカルのモデル/データディレクトリ (Streamlit Cloudなどデプロイ環境では読み取り専用か、存在しない可能性あり)
MODELS_DIR = Path("models")
DATA_DIR = Path("data/modified")

# GitHubからall_stock_model_performance.jsonをロードするためのURL
# yuutaka69/kabuka_agaru リポジトリの main ブランチの models/all_stock_model_performance.json
GITHUB_RAW_URL_BASE = "https://raw.githubusercontent.com/yuutaka69/kabuka_agaru/main/"
ALL_PERFORMANCE_JSON_URL = f"{GITHUB_RAW_URL_BASE}models/all_stock_model_performance.json"

# --- 関数: モデルとデータのロード ---

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
        
        # joblib.load はファイルパスまたはファイルライクオブジェクトを期待するので、BytesIOを使う
        from io import BytesIO
        loaded_content = joblib.load(BytesIO(response.content))

        if isinstance(loaded_content, tuple) and len(loaded_content) == 2:
            model, feature_names = loaded_content
            st.success(f"モデル '{model_filename}' をGitHubからロードしました。特徴量数: {len(feature_names)}")
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
        st.success(f"データ '{data_filename}' をGitHubからロードしました。行数: {len(df)}")
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"エラー: GitHubからデータ '{data_filename}' をダウンロードできませんでした: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"エラー: データファイルのロード中に問題が発生しました: {e}")
        return pd.DataFrame()

@st.cache_data
def load_all_performance_data_from_github():
    """
    GitHubから全体のモデル性能JSONデータをロードする。
    """
    try:
        response = requests.get(ALL_PERFORMANCE_JSON_URL)
        response.raise_for_status() # HTTPエラーを確認
        perf_data = response.json()
        st.sidebar.success(f"モデル性能データ '{Path(ALL_PERFORMANCE_JSON_URL).name}' をGitHubからロードしました。")
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

st.set_page_config(layout="wide", page_title="株価予測ダッシュボード")

st.title("📈 株価予測ダッシュボード")
st.markdown("このアプリは、**GitHubに保存された学習済みモデルと評価結果**を使用して、過去のテクニカル指標に基づいた将来の株価（指定された期間での上昇）予測を表示します。")

st.sidebar.header("アプリ情報")
st.sidebar.write(f"データソース: GitHub ({GITHUB_RAW_URL_BASE.split('//')[1].split('/')[0]})")
st.sidebar.write(f"LightGBMバージョン: `{lgb.__version__}`")
st.sidebar.markdown("---")

# 全体のモデル性能データを最初にロード
all_performance_data = load_all_performance_data_from_github()

if all_performance_data is None:
    st.error("アプリケーションを続行できません。GitHubからモデル性能データをロードできませんでした。")
    st.stop()

st.header("銘柄と予測期間の選択")

# 利用可能な銘柄とターゲット期間のリストをall_performance_dataから動的に取得
available_stocks = sorted(list(all_performance_data.keys()))

if not available_stocks:
    st.error("GitHubからロードされたモデル性能データに銘柄情報が見つかりません。")
    st.stop()

selected_stock = st.selectbox(
    "予測したい銘柄を選択してください:",
    options=available_stocks,
    index=0
)

available_target_periods = []
if selected_stock in all_performance_data:
    available_target_periods = sorted([int(p.replace('d', '')) for p in all_performance_data[selected_stock].keys()])

if not available_target_periods:
    st.warning(f"銘柄 '{selected_stock}' に対応する予測期間が見つかりません。")
    st.stop()

selected_target_period = st.selectbox(
    "予測したいターゲット期間 (N日後の上昇) を選択してください:",
    options=available_target_periods,
    index=available_target_periods.index(120) if 120 in available_target_periods else 0
)

# 選択された銘柄とターゲット期間に対応するモデルとデータをロード
model, trained_feature_names = load_model_and_features_from_github(selected_stock, selected_target_period)
df_selected_stock = load_stock_data_from_github(selected_stock)

if model is None or trained_feature_names is None or df_selected_stock.empty:
    st.error("必要なモデルまたはデータをロードできませんでした。予測を実行できません。")
    st.stop()

st.header("予測実行")

features_for_prediction = []
if trained_feature_names is not None and not df_selected_stock.empty:
    features_for_prediction = [col for col in trained_feature_names if col in df_selected_stock.columns]
    missing_features = [col for col in trained_feature_names if col not in df_selected_stock.columns]
    if missing_features:
        st.warning(f"選択された銘柄のデータに、モデルが訓練された際の特徴量の一部が見つかりません: {', '.join(missing_features)}。予測精度に影響する可能性があります。")
    if not features_for_prediction:
        st.error("モデルが期待する特徴量と一致する列が、選択された銘柄のデータにありません。予測できません。")
        st.stop()
elif trained_feature_names is None:
    st.error("モデルから特徴量リストが取得できませんでした。予測できません。")
    st.stop()
elif df_selected_stock.empty:
    st.error("銘柄データがロードされていません。予測できません。")
    st.stop()

# 予測対象日の選択
available_dates = df_selected_stock.index.unique().sort_values(ascending=False)

if available_dates.empty:
    st.warning("選択された銘柄のデータに有効な日付がありません。予測を実行できません。")
    st.stop()

default_date_index = 0
selected_date = st.selectbox(
    "予測を行いたい日付を選択してください (この日のデータで予測します):",
    options=available_dates.date,
    index=default_date_index
)

# 選択された日付のデータを取得し、NaNを処理
prediction_data_series = df_selected_stock.loc[pd.to_datetime(selected_date)]

# 訓練時の特徴量とデータフレームのカラムを比較し、一致するように調整
X_predict = pd.DataFrame(columns=trained_feature_names) # 訓練時のカラムで空のDFを作成
# 既存のデータをコピーし、不足するカラムをNaNで埋める
temp_df = pd.DataFrame([prediction_data_series.reindex(features_for_prediction).values], columns=features_for_prediction)
for col in trained_feature_names:
    if col in temp_df.columns:
        X_predict[col] = temp_df[col]
    else:
        X_predict[col] = np.nan # 訓練時にはあったが現在のデータにはない場合、NaNで埋める

X_predict.replace([np.inf, -np.inf], np.nan, inplace=True) # NaN/infのクリーンアップ

st.subheader(f"選択された銘柄 ({selected_stock}) - {selected_date} のデータ概要:")
display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
indicator_display_cols = [f for f in features_for_prediction if any(f.startswith(prefix) for prefix in ['SMA', 'RSI', 'MACD', 'stoch', 'BBANDS', 'adx', 'atr', 'roc'])]
display_cols.extend(indicator_display_cols[:10]) # 上位10個の指標を表示
existing_display_cols = [col for col in display_cols if col in prediction_data_series.keys()]

if not existing_display_cols:
    st.warning("表示可能なデータ列がありません。")
else:
    st.dataframe(prediction_data_series[existing_display_cols].to_frame().T)
st.markdown("---")

# 予測実行ボタン
if st.button("📈 株価上昇を予測する"):
    with st.spinner("予測を実行中..."):
        if X_predict.isnull().any().any():
            st.warning("選択された日付のデータにNaNが含まれています。これらのNaNは予測前にモデルによって処理されますが、精度に影響する可能性があります。")
        
        prediction_proba = model.predict_proba(X_predict)[:, 1][0]
        prediction_class = model.predict(X_predict)[0]

        st.subheader("予測結果")
        if prediction_class == 1:
            st.success(f"**ポジティブなシグナル: 株価が{selected_target_period}日以内に上昇する可能性が高いです！**")
            st.metric(f"{selected_target_period}日後の上昇確率", f"{prediction_proba:.2%}", delta_color="off")
        else:
            st.info(f"**ニュートラル/ネガティブなシグナル: 株価が{selected_target_period}日以内に上昇する可能性は低いかもしれません。**")
            st.metric(f"{selected_target_period}日後の上昇確率", f"{prediction_proba:.2%}", delta_color="off")

        st.markdown("---")
        st.subheader("モデルのパフォーマンス指標")
        
        stock_perf_data = all_performance_data.get(selected_stock, {})
        model_perf_data = stock_perf_data.get(f"{selected_target_period}d", {})

        if model_perf_data:
            st.write(f"#### 訓練時評価 ({selected_target_period}日予測モデル)")
            training_metrics = model_perf_data.get('training_evaluation', {})
            if training_metrics:
                st.write(f"- **テストセットでの精度 (Accuracy):** `{training_metrics.get('accuracy', 'N/A'):.2f}`")
                st.write(f"- **ROC AUC スコア:** `{training_metrics.get('roc_auc_score', 'N/A'):.2f}`")
                st.markdown(f"**クラス 1 (上昇する) のメトリクス:**")
                class_1_train = training_metrics.get('class_1_metrics', {})
                st.write(f"- 適合率 (Precision): `{class_1_train.get('precision', 'N/A'):.2f}`")
                st.write(f"- 再現率 (Recall): `{class_1_train.get('recall', 'N/A'):.2f}`")
                st.write(f"- F1スコア: `{class_1_train.get('f1-score', 'N/A'):.2f}`")
                st.caption("※これらの指標はモデルが訓練・評価された際のテストデータに基づいています。")
            else:
                st.warning("訓練時評価データが見つかりませんでした。")
            
            st.write(f"#### 直近データ評価 ({selected_target_period}日予測モデル)")
            recent_metrics = model_perf_data.get('recent_data_evaluation', {})
            if recent_metrics:
                st.write(f"- **評価期間 (日数):** `{recent_metrics.get('total_evaluated_days', 'N/A')}`")
                st.write(f"- **精度 (Accuracy):** `{recent_metrics.get('accuracy', 'N/A'):.2f}`")
                st.write(f"- **ROC AUC スコア:** `{recent_metrics.get('roc_auc_score', 'N/A'):.2f}`")
                st.markdown(f"**クラス 1 (上昇する) のメトリクス:**")
                class_1_recent = recent_metrics.get('class_1_metrics', {}) # all_stock_model_performance.json に class_1_metrics がない場合は、個別のキーを参照
                st.write(f"- 適合率 (Precision): `{recent_metrics.get('precision_class_1', 'N/A'):.2f}`")
                st.write(f"- 再現率 (Recall): `{recent_metrics.get('recall_class_1', 'N/A'):.2f}`")
                st.write(f"- F1スコア: `{recent_metrics.get('f1_score_class_1', 'N/A'):.2f}`")
                st.write(f"- **直近最終日の予測日:** `{recent_metrics.get('most_recent_prediction_date', 'N/A')}`")
                st.write(f"- **直近最終日の予測結果:** `{recent_metrics.get('most_recent_prediction_value', 'N/A')}` (`{recent_metrics.get('most_recent_prediction_proba', 'N/A'):.2%}`)")
                st.caption("※これらの指標はモデルが評価された直近の期間データに基づいています。")
            else:
                st.warning("直近データ評価が見つかりませんでした。")
        else:
            st.warning("このモデルの性能データが全体JSONに見つかりませんでした。")

        st.markdown("---")
        st.subheader("予測に影響した特徴量")
        if hasattr(model, 'feature_importances_') and trained_feature_names is not None:
            feature_importance = pd.Series(model.feature_importances_, index=trained_feature_names)
            top_features = feature_importance.nlargest(15)
            st.bar_chart(top_features)
        else:
            st.warning("モデルから特徴量重要度を取得できませんでした。")

        st.markdown("---")
        st.info("※ この予測はモデルに基づいたものであり、将来の株価を保証するものではありません。投資は自己責任でお願いします。")

# --- サイドバーに全体ランキングを表示 ---
st.sidebar.header("全モデル性能ランキング")

if all_performance_data:
    ranking_data = []
    for stock_code, periods_data in all_performance_data.items():
        for period_str, model_data in periods_data.items():
            training_eval = model_data.get('training_evaluation', {})
            recent_eval = model_data.get('recent_data_evaluation', {})

            # 直近データ評価のF1スコアを優先的に使用、なければ訓練時のF1スコア
            f1_score_recent_class1 = recent_eval.get('f1_score_class_1', np.nan)
            if pd.isna(f1_score_recent_class1): # 直近データ評価がなければ訓練時評価から
                f1_score_recent_class1 = training_eval.get('class_1_metrics', {}).get('f1-score', np.nan)

            ranking_data.append({
                '銘柄': stock_code,
                '期間': int(period_str.replace('d', '')),
                '訓練Acc': training_eval.get('accuracy', np.nan),
                '訓練AUC': training_eval.get('roc_auc_score', np.nan),
                '訓練F1(1)': training_eval.get('class_1_metrics', {}).get('f1-score', np.nan),
                '直近Acc': recent_eval.get('accuracy', np.nan),
                '直近AUC': recent_eval.get('roc_auc_score', np.nan),
                '直近F1(1)': recent_eval.get('f1_score_class_1', np.nan),
                '最終予測日': recent_eval.get('most_recent_prediction_date', 'N/A'),
                '最終予測': recent_eval.get('most_recent_prediction_value', 'N/A'),
                '最終予測確率': recent_eval.get('most_recent_prediction_proba', np.nan)
            })

    if ranking_data:
        df_ranking = pd.DataFrame(ranking_data)
        df_ranking.fillna('N/A', inplace=True) # NaNを'N/A'に置換して表示をきれいに

        # ソートオプション
        sort_by_options = {
            "直近F1(1) (降順)": "直近F1(1)",
            "訓練F1(1) (降順)": "訓練F1(1)",
            "直近Acc (降順)": "直近Acc",
            "訓練Acc (降順)": "訓練Acc",
            "銘柄 (昇順)": "銘柄"
        }
        sort_by = st.sidebar.selectbox("ソート基準:", list(sort_by_options.keys()))
        
        ascending = False if "降順" in sort_by else True
        sort_column = sort_by_options[sort_by]

        # 数値カラムを適切にソートするために、N/AをNaNに戻してソート
        if sort_column in ['訓練Acc', '訓練AUC', '訓練F1(1)', '直近Acc', '直近AUC', '直近F1(1)', '最終予測確率']:
            df_ranking[sort_column] = pd.to_numeric(df_ranking[sort_column], errors='coerce')
            df_ranking_sorted = df_ranking.sort_values(by=sort_column, ascending=ascending, na_position='last')
        else:
            df_ranking_sorted = df_ranking.sort_values(by=sort_column, ascending=ascending)

        st.sidebar.dataframe(df_ranking_sorted.head(20).style.format({
            '訓練Acc': "{:.2f}", '訓練AUC': "{:.2f}", '訓練F1(1)': "{:.2f}",
            '直近Acc': "{:.2f}", '直近AUC': "{:.2f}", '直近F1(1)': "{:.2f}",
            '最終予測確率': "{:.2%}"
        }))
        st.sidebar.caption("※ 上位20件を表示")
    else:
        st.sidebar.warning("ランキングデータを生成できませんでした。")

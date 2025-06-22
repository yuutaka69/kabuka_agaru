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

# --- 設定 ---
MODELS_DIR = Path("models") # モデルとメトリクスJSONの共通ディレクトリと仮定
DATA_DIR = Path("data/modified")

# --- 関数: モデルとデータのロード ---

@st.cache_resource
def load_model_and_features(model_path):
    """
    指定されたパスからLightGBMモデルと訓練時に使用された特徴量リストをロードする。
    """
    if not model_path.exists():
        st.error(f"エラー: モデルファイルが見つかりません。パスを確認してください: `{model_path}`")
        st.stop()
    try:
        loaded_content = joblib.load(model_path)
        if isinstance(loaded_content, tuple) and len(loaded_content) == 2:
            model, feature_names = loaded_content
            st.success(f"モデル '{model_path.name}' をロードしました。特徴量数: {len(feature_names)}")
            return model, feature_names
        else:
            st.error(f"エラー: モデルファイル '{model_path.name}' の形式が不正です。モデルと特徴量リストがセットで保存されていません。")
            st.stop()
    except Exception as e:
        st.error(f"エラー: モデルのロード中に問題が発生しました: {e}")
        st.stop()

@st.cache_data
def load_stock_data(data_filepath):
    """
    指定されたCSVファイルをロードし、LightGBMが扱えるように前処理を行う。
    """
    if not data_filepath.exists():
        st.error(f"エラー: データファイルが見つかりません。パスを確認してください: `{data_filepath}`")
        st.stop()
    try:
        df = pd.read_csv(data_filepath, index_col='Date', parse_dates=True)
        df.dropna(axis='columns', how='all', inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        st.success(f"データ '{data_filepath.name}' をロードしました。行数: {len(df)}")
        return df
    except Exception as e:
        st.error(f"エラー: データファイルのロード中に問題が発生しました: {e}")
        st.stop()

@st.cache_data
def load_model_performance(perf_filepath):
    """
    指定されたJSONファイルからモデル性能データをロードする。
    """
    if not perf_filepath.exists():
        st.warning(f"警告: モデル性能ファイルが見つかりません。パス: `{perf_filepath}`")
        return None
    try:
        with open(perf_filepath, 'r') as f:
            perf_data = json.load(f)
        st.success(f"モデル性能データ '{perf_filepath.name}' をロードしました。")
        return perf_data
    except json.JSONDecodeError:
        st.error(f"エラー: モデル性能ファイル '{perf_filepath.name}' のJSON形式が不正です。")
        return None
    except Exception as e:
        st.error(f"エラー: モデル性能のロード中に問題が発生しました: {e}")
        return None

# --- Streamlit UIの構築 ---

st.title("📈 株価予測ダッシュボード")
st.markdown("このアプリは、過去のテクニカル指標に基づいて将来の株価（指定された期間での上昇）を予測します。")

st.sidebar.header("アプリ情報")
st.sidebar.write(f"モデル/メトリクスディレクトリ: `{MODELS_DIR}`")
st.sidebar.write(f"データディレクトリ: `{DATA_DIR}`")
st.sidebar.write(f"LightGBMバージョン: `{lgb.__version__}`")
st.sidebar.markdown("---")

st.header("銘柄と予測期間の選択")

# 1. 利用可能な銘柄リストの取得
available_stocks = []
if DATA_DIR.exists():
    available_data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_modified.csv')]
    if available_data_files:
        available_stocks = sorted([f.replace('_modified.csv', '') for f in available_data_files])
    else:
        st.warning(f"データディレクトリ '{DATA_DIR}' にCSVファイルが見つかりません。")
else:
    st.error(f"エラー: データディレクトリ '{DATA_DIR}' が見つかりません。アプリケーションをデプロイする前に、`data/modified` フォルダが存在し、データファイルが含まれていることを確認してください。")
    st.stop()

selected_stock = st.selectbox(
    "予測したい銘柄を選択してください:",
    options=available_stocks,
    index=0 if available_stocks else None
)

df_selected_stock = pd.DataFrame()
if selected_stock:
    selected_data_filepath = DATA_DIR / f"{selected_stock}_modified.csv"
    df_selected_stock = load_stock_data(selected_data_filepath)
else:
    st.warning("選択できる銘柄がありません。")
    st.stop()

# 2. 利用可能なターゲット期間の取得
available_target_periods = []
if MODELS_DIR.exists():
    # モデルファイルとメトリクスファイルの両方から期間を抽出するロジックに変更
    available_model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"lgbm_model_{selected_stock}_") and f.endswith('.joblib')]
    available_metrics_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"{selected_stock}_") and f.endswith('_metrics.json')] # 新しい命名規則に対応

    # joblibファイルから期間を抽出
    periods_from_models = {
        int(f.replace(f"lgbm_model_{selected_stock}_", "").replace("d.joblib", ""))
        for f in available_model_files
        if f.replace(f"lgbm_model_{selected_stock}_", "").replace("d.joblib", "").isdigit()
    }
    
    # metrics JSONファイルから期間を抽出 (例: 1547_120d_metrics.json -> 120)
    periods_from_metrics = {
        int(f.replace(f"{selected_stock}_", "").replace("d_metrics.json", ""))
        for f in available_metrics_files
        if f.replace(f"{selected_stock}_", "").replace("d_metrics.json", "").isdigit()
    }

    # 両方のファイルに存在する期間のみを対象とする
    available_target_periods = sorted(list(periods_from_models.intersection(periods_from_metrics)))

    if not available_target_periods:
        st.warning(f"銘柄 '{selected_stock}' に対応する有効なモデルとメトリクスファイルが見つかりません。")
else:
    st.error(f"エラー: モデルディレクトリ '{MODELS_DIR}' が見つかりません。アプリケーションをデプロイする前に、`models` フォルダが存在し、ファイルが含まれていることを確認してください。")
    st.stop()

model = None
trained_feature_names = None
model_performance_data = None

if available_target_periods:
    selected_target_period = st.selectbox(
        "予測したいターゲット期間 (N日後の上昇) を選択してください:",
        options=available_target_periods,
        index=available_target_periods.index(120) if 120 in available_target_periods else 0
    )

    # 選択された銘柄とターゲット期間に対応するモデルのパスを構築
    selected_model_filename = f"lgbm_model_{selected_stock}_{selected_target_period}d.joblib"
    selected_model_path = MODELS_DIR / selected_model_filename
    
    # モデルと訓練時の特徴量リストをロード
    model, trained_feature_names = load_model_and_features(selected_model_path)

    # モデル性能データをロード (新しい命名規則に対応)
    selected_perf_filename = f"{selected_stock}_{selected_target_period}d_metrics.json" # サンプルJSONの命名規則に合わせる
    selected_perf_filepath = MODELS_DIR / selected_perf_filename # MODELS_DIR にあると仮定
    model_performance_data = load_model_performance(selected_perf_filepath)

else:
    st.warning("選択できるターゲット期間がありません。")
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

prediction_data_series = df_selected_stock.loc[pd.to_datetime(selected_date)]
X_predict = pd.DataFrame([prediction_data_series[features_for_prediction].values], columns=features_for_prediction)

st.subheader(f"選択された銘柄 ({selected_stock}) - {selected_date} のデータ概要:")
display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
indicator_display_cols = [f for f in features_for_prediction if any(f.startswith(prefix) for prefix in ['SMA', 'RSI', 'MACD', 'stoch', 'BBANDS'])]
display_cols.extend(indicator_display_cols[:5])
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
        if model_performance_data:
            st.write("---")
            st.write(f"**総合評価:**")
            st.write(f"- **テストセットでの精度 (Accuracy):** `{model_performance_data.get('accuracy', 'N/A'):.2f}`")
            st.write(f"- **ROC AUC スコア:** `{model_performance_data.get('roc_auc_score', 'N/A'):.2f}`")
            st.write("---")
            st.write(f"**クラスごとの詳細 (0: 上昇しない, 1: 上昇する):**")
            
            # クラス0のメトリクス
            class_0 = model_performance_data.get('class_0_metrics', {})
            st.markdown(f"**クラス 0 (上昇しない) のメトリクス:**")
            st.write(f"- 適合率 (Precision): `{class_0.get('precision', 'N/A'):.2f}`")
            st.write(f"- 再現率 (Recall): `{class_0.get('recall', 'N/A'):.2f}`")
            st.write(f"- F1スコア: `{class_0.get('f1-score', 'N/A'):.2f}`")
            st.write(f"- サポート数: `{int(class_0.get('support', 0))}`") # supportは整数で表示

            st.markdown(f"**クラス 1 (上昇する) のメトリクス:**")
            class_1 = model_performance_data.get('class_1_metrics', {})
            st.write(f"- 適合率 (Precision): `{class_1.get('precision', 'N/A'):.2f}`")
            st.write(f"- 再現率 (Recall): `{class_1.get('recall', 'N/A'):.2f}`")
            st.write(f"- F1スコア: `{class_1.get('f1-score', 'N/A'):.2f}`")
            st.write(f"- サポート数: `{int(class_1.get('support', 0))}`") # supportは整数で表示

            st.markdown("---")
            st.write("**混同行列:**")
            cm = model_performance_data.get('confusion_matrix', [[0,0],[0,0]])
            st.dataframe(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))
            
            st.caption("※これらの指標はモデルが訓練・評価された際のテストデータに基づいています。")
            st.caption("※適合率(Precision)は「陽性と予測した中で実際に陽性だった割合」、再現率(Recall)は「実際に陽性だったものをどれだけ陽性と予測できたか」を示します。")
            st.caption("※ROC AUCスコアはモデルの分類性能のバランスを示す指標です。")
        else:
            st.warning("このモデルの性能データが見つかりませんでした。")

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

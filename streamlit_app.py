import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import lightgbm as lgb # モデルのバージョン互換性のため

# --- 設定 ---
# GitHubリポジトリのルートを基準としたパス
# Streamlit Community Cloudでデプロイする場合、これらのパスはGitHubリポジトリの構造と一致する必要があります。
MODELS_DIR = "models"
DATA_DIR = "data/modified"

# --- 関数: モデルとデータのロード ---

@st.cache_resource # モデルのロードはアプリ起動時に一度だけ行うようにキャッシュ
def load_model_and_features(model_path):
    """
    指定されたパスからLightGBMモデルと訓練時に使用された特徴量リストをロードする。
    """
    try:
        # 訓練時にモデルと特徴量リストをセットで保存していることを想定
        loaded_content = joblib.load(model_path)
        if isinstance(loaded_content, tuple) and len(loaded_content) == 2:
            model, feature_names = loaded_content
            st.success(f"モデル '{os.path.basename(model_path)}' をロードしました。特徴量数: {len(feature_names)}")
            return model, feature_names
        else:
            st.error(f"エラー: モデルファイル '{os.path.basename(model_path)}' の形式が不正です。モデルと特徴量リストがセットで保存されていません。")
            st.stop()
    except FileNotFoundError:
        st.error(f"エラー: モデルファイルが見つかりません。パスを確認してください: {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"エラー: モデルのロード中に問題が発生しました: {e}")
        st.stop()

@st.cache_data # データのロードもキャッシュ (銘柄ごとにキャッシュ)
def load_stock_data(data_filepath):
    """
    指定されたCSVファイルをロードし、LightGBMが扱えるように前処理を行う。
    """
    try:
        df = pd.read_csv(data_filepath, index_col='Date', parse_dates=True)
        df.dropna(axis='columns', how='all', inplace=True) # 空の列を削除
        df.replace([np.inf, -np.inf], np.nan, inplace=True) # 無限大をNaNに
        st.success(f"データ '{os.path.basename(data_filepath)}' をロードしました。行数: {len(df)}")
        return df
    except FileNotFoundError:
        st.error(f"エラー: データファイル '{data_filepath}' が見つかりません。パスを確認してください。")
        st.stop()
    except Exception as e:
        st.error(f"エラー: データファイルのロード中に問題が発生しました: {e}")
        st.stop()

# --- Streamlit UIの構築 ---

st.title("📈 株価予測ダッシュボード")
st.markdown("このアプリは、過去のテクニカル指標に基づいて将来の株価（指定された期間での上昇）を予測します。")

st.sidebar.header("アプリ情報")
st.sidebar.write(f"モデルディレクトリ: `{MODELS_DIR}`")
st.sidebar.write(f"データディレクトリ: `{DATA_DIR}`")
st.sidebar.write(f"LightGBMバージョン: `{lgb.__version__}`")
st.sidebar.markdown("---")

st.header("銘柄と予測期間の選択")

# 1. 利用可能な銘柄リストの取得
try:
    # data/modified ディレクトリからCSVファイルをスキャン
    available_data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_modified.csv')]
    if not available_data_files:
        st.error(f"エラー: データディレクトリ '{DATA_DIR}' にCSVファイルが見つかりません。")
        st.stop()
    
    # ファイル名から銘柄コードを抽出 (例: QQQ_modified.csv -> QQQ)
    available_stocks = sorted([f.replace('_modified.csv', '') for f in available_data_files])
    
    selected_stock = st.selectbox(
        "予測したい銘柄を選択してください:",
        options=available_stocks,
        index=0 if available_stocks else None # デフォルトで最初の銘柄を選択
    )

    if not selected_stock:
        st.warning("選択できる銘柄がありません。")
        st.stop()

    # 選択された銘柄のデータをロード
    selected_data_filepath = os.path.join(DATA_DIR, f"{selected_stock}_modified.csv")
    df_selected_stock = load_stock_data(selected_data_filepath)

except FileNotFoundError:
    st.error(f"エラー: データディレクトリ '{DATA_DIR}' が見つかりません。")
    st.stop()
except Exception as e:
    st.error(f"エラー: 銘柄リストの取得またはデータロード中に問題が発生しました: {e}")
    st.stop()

# 2. 利用可能なターゲット期間の取得
try:
    # models/ ディレクトリからモデルファイルをスキャンし、ターゲット期間を抽出
    available_model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"lgbm_model_{selected_stock}_") and f.endswith('.joblib')]
    if not available_model_files:
        st.error(f"エラー: 銘柄 '{selected_stock}' に対応するモデルファイルが '{MODELS_DIR}' に見つかりません。")
        st.stop()
    
    # モデルファイル名からターゲット期間を抽出 (例: lgbm_model_QQQ_14d.joblib -> 14)
    available_target_periods = sorted([
        int(f.replace(f"lgbm_model_{selected_stock}_", "").replace("d.joblib", ""))
        for f in available_model_files
        if f.replace(f"lgbm_model_{selected_stock}_", "").replace("d.joblib", "").isdigit() # 数字部分のみを抽出
    ])
    
    if not available_target_periods:
        st.error(f"エラー: 銘柄 '{selected_stock}' に対応する有効なターゲット期間を持つモデルファイルが見つかりません。")
        st.stop()

    selected_target_period = st.selectbox(
        "予測したいターゲット期間 (N日後の上昇) を選択してください:",
        options=available_target_periods,
        index=available_target_periods.index(120) if 120 in available_target_periods else 0 # デフォルトで120日を選択 (あれば)
    )

    # 選択された銘柄とターゲット期間に対応するモデルのパスを構築
    selected_model_filename = f"lgbm_model_{selected_stock}_{selected_target_period}d.joblib"
    selected_model_path = os.path.join(MODELS_DIR, selected_model_filename)
    
    # モデルと訓練時の特徴量リストをロード
    model, trained_feature_names = load_model_and_features(selected_model_path)

except FileNotFoundError:
    st.error(f"エラー: モデルディレクトリ '{MODELS_DIR}' が見つかりません。")
    st.stop()
except Exception as e:
    st.error(f"エラー: ターゲット期間の取得またはモデルロード中に問題が発生しました: {e}")
    st.stop()


st.header("予測実行")

# モデルに渡す特徴量列を決定
# モデルから取得した特徴量リストを使用するのが最も堅牢
if trained_feature_names is not None:
    features_for_prediction = [col for col in trained_feature_names if col in df_selected_stock.columns]
    missing_features = [col for col in trained_feature_names if col not in df_selected_stock.columns]
    if missing_features:
        st.warning(f"選択された銘柄のデータに、モデルが訓練された際の特徴量の一部が見つかりません: {', '.join(missing_features)}。予測精度に影響する可能性があります。")
    if not features_for_prediction:
        st.error("モデルが期待する特徴量と一致する列が、選択された銘柄のデータにありません。予測できません。")
        st.stop()
else:
    st.error("モデルから特徴量リストが取得できませんでした。予測できません。")
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

# 選択された日付のデータを取得し、予測に使用する特徴量のみを抽出
prediction_data_series = df_selected_stock.loc[pd.to_datetime(selected_date)]
X_predict = pd.DataFrame([prediction_data_series[features_for_prediction].values], columns=features_for_prediction)

st.subheader(f"選択された銘柄 ({selected_stock}) - {selected_date} のデータ概要:")
# 主要なOHLCVと一部の指標を表示
display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
indicator_display_cols = [f for f in features_for_prediction if any(f.startswith(prefix) for prefix in ['SMA', 'RSI', 'MACD', 'stoch'])]
display_cols.extend(indicator_display_cols)
existing_display_cols = [col for col in display_cols if col in prediction_data_series.index] # `in prediction_data_series.index` は間違い、`in prediction_data_series` または `col in prediction_data_series.keys()`
existing_display_cols = [col for col in display_cols if col in prediction_data_series.keys()]


if not existing_display_cols:
    st.warning("表示可能なデータ列がありません。")
else:
    st.dataframe(prediction_data_series[existing_display_cols].to_frame().T)
st.markdown("---")

# 予測実行ボタン
if st.button("📈 株価上昇を予測する"):
    with st.spinner("予測を実行中..."):
        # NaN値があるか確認
        if X_predict.isnull().all().all():
            st.error("選択された日付のデータが全てNaNのため、予測できません。")
        else:
            prediction_proba = model.predict_proba(X_predict)[:, 1][0]
            prediction_class = model.predict(X_predict)[0]

            st.subheader("予測結果")
            if prediction_class == 1:
                st.success(f"**ポジティブなシグナル: 株価が{selected_target_period}日以内に上昇する可能性が高いです！**")
                st.metric(f"{selected_target_period}日以内の上昇確率", f"{prediction_proba:.2%}", delta_color="off")
            else:
                st.info(f"**ニュートラル/ネガティブなシグナル: 株価が{selected_target_period}日以内に上昇しない可能性が高いです。**")
                st.metric(f"{selected_target_period}日以内の上昇確率", f"{prediction_proba:.2%}", delta_color="off")

            st.markdown("---")
            st.subheader("予測に影響した特徴量")
            if hasattr(model, 'feature_importances_') and model.feature_name_ is not None:
                feature_importance = pd.Series(model.feature_importances_, index=model.feature_name_)
                top_features = feature_importance[features_for_prediction].nlargest(15)
                st.bar_chart(top_features)
            else:
                st.warning("モデルから特徴量重要度を取得できませんでした。")

            st.markdown("---")
            st.info("※ この予測はモデルに基づいたものであり、将来の株価を保証するものではありません。投資は自己責任でお願いします。")
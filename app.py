import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os

from src.components.data_loader import DataLoader
from src.components.data_analyzer import DataAnalyzer
from src.components.model_builder import ModelBuilder
from src.components.feature_selector import FeatureSelector
from src.utils.config import MAX_COLUMNS_BEFORE_SELECTION, SUPPORTED_FILE_TYPES, MAX_FILE_SIZE_MB

# ページ設定
st.set_page_config(
    page_title="機械学習自動適用アプリ",
    page_icon="🤖",
    layout="wide"
)

# タイトル
st.title("🤖 機械学習自動適用アプリ")
st.markdown("データ分析から機械学習モデルの構築・評価までを自動化")

# セッション状態の初期化
if 'data' not in st.session_state:
    st.session_state.data = None
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None


def plot_missing_values(missing_info: pd.DataFrame):
    """欠損値の可視化"""
    if len(missing_info) == 0:
        st.info("欠損値はありません")
        return
    
    fig = px.bar(
        missing_info, 
        x='column', 
        y='missing_percentage',
        title='欠損値の割合 (%)',
        labels={'missing_percentage': '欠損率 (%)', 'column': 'カラム名'}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """相関行列のヒートマップ"""
    if corr_matrix.empty:
        st.info("数値データがありません")
        return
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='相関行列ヒートマップ',
        xaxis_title='',
        yaxis_title='',
        width=800,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_distributions(df: pd.DataFrame, numeric_cols: list):
    """数値データの分布を可視化"""
    if not numeric_cols:
        st.info("数値データがありません")
        return
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) - 1) // n_cols + 1
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=numeric_cols[:n_rows*n_cols]
    )
    
    for idx, col in enumerate(numeric_cols[:n_rows*n_cols]):
        row = idx // n_cols + 1
        col_idx = idx % n_cols + 1
        
        data = df[col].dropna()
        fig.add_trace(
            go.Histogram(x=data, name=col, showlegend=False),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        title='数値データの分布',
        height=300 * n_rows,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_outliers(df: pd.DataFrame, outliers: dict):
    """外れ値の可視化（箱ひげ図）"""
    numeric_cols = list(outliers.keys())
    if not numeric_cols:
        st.info("外れ値が検出されませんでした")
        return
    
    # 標準化オプション
    standardize_option = st.radio(
        "表示方法を選択",
        ["元の値", "標準化（Zスコア）", "正規化（0-1）"],
        horizontal=True
    )
    
    # データの準備
    plot_data = df[numeric_cols].copy()
    
    if standardize_option == "標準化（Zスコア）":
        # 標準化（平均0、標準偏差1）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        plot_data[numeric_cols] = scaler.fit_transform(plot_data[numeric_cols])
        y_title = '標準化された値（Zスコア）'
        title = '外れ値の可視化（標準化後）'
    elif standardize_option == "正規化（0-1）":
        # 正規化（0-1の範囲）
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        plot_data[numeric_cols] = scaler.fit_transform(plot_data[numeric_cols])
        y_title = '正規化された値（0-1）'
        title = '外れ値の可視化（正規化後）'
    else:
        y_title = '値'
        title = '外れ値の可視化（元の値）'
    
    # 個別のサブプロットで表示するオプション
    display_option = st.checkbox("個別のグラフで表示", value=False)
    
    if display_option:
        # サブプロットで表示
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) - 1) // n_cols + 1
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=numeric_cols,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols + 1
            col_idx = idx % n_cols + 1
            
            fig.add_trace(
                go.Box(
                    y=plot_data[col],
                    name=col,
                    boxpoints='outliers',
                    showlegend=False
                ),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            title=title,
            height=300 * n_rows,
            showlegend=False
        )
    else:
        # 単一のグラフで表示
        fig = go.Figure()
        
        for col in numeric_cols:
            fig.add_trace(go.Box(
                y=plot_data[col],
                name=col,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title=y_title,
            showlegend=True
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 外れ値の統計情報を表示
    if standardize_option != "元の値":
        st.info(f"""
        **{standardize_option}について**
        - {"標準化: 各変数の平均を0、標準偏差を1に変換" if standardize_option == "標準化（Zスコア）" else "正規化: 各変数を0〜1の範囲に変換"}
        - すべての変数が同じスケールになるため、比較が容易になります
        - {"±2〜3の範囲外の値が外れ値の可能性が高い" if standardize_option == "標準化（Zスコア）" else "0.1未満や0.9を超える値に注目"}
        """)


# タブ作成
tab1, tab2, tab3 = st.tabs(["📊 データプレビュー", "🔍 自動分析", "🤖 機械学習モデル構築"])

# データプレビュータブ
with tab1:
    st.header("📁 データアップロード")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "ファイルを選択してください",
        type=SUPPORTED_FILE_TYPES,
        help=f"対応形式: {', '.join(SUPPORTED_FILE_TYPES)} (最大{MAX_FILE_SIZE_MB}MB)"
    )
    
    if uploaded_file is not None:
        # ファイルサイズチェック
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"ファイルサイズが大きすぎます。{MAX_FILE_SIZE_MB}MB以下のファイルを選択してください。")
        else:
            # 一時ファイルとして保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # データ読み込み
                loader = DataLoader()
                data = loader.load_file(tmp_path)
                
                # カラム選択が必要かチェック
                if loader.needs_column_selection(data):
                    st.warning(f"カラム数が{MAX_COLUMNS_BEFORE_SELECTION}を超えています。分析対象のカラムを選択してください。")
                    
                    selected_cols = st.multiselect(
                        "分析対象のカラムを選択",
                        options=data.columns.tolist(),
                        default=data.columns[:MAX_COLUMNS_BEFORE_SELECTION].tolist()
                    )
                    
                    if st.button("選択したカラムで分析開始"):
                        st.session_state.data = loader.select_columns(data, selected_cols)
                        st.session_state.selected_columns = selected_cols
                        st.success(f"{len(selected_cols)}個のカラムが選択されました")
                else:
                    st.session_state.data = data
                    st.success("データが正常に読み込まれました")
                
                # ファイル情報表示
                if st.session_state.data is not None:
                    file_info = loader.get_file_info(st.session_state.data)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("行数", f"{file_info['rows']:,}")
                    with col2:
                        st.metric("列数", f"{file_info['columns']:,}")
                    with col3:
                        st.metric("メモリ使用量", f"{file_info['memory_usage'] / 1024 / 1024:.2f} MB")
                    
            except Exception as e:
                st.error(f"ファイルの読み込みに失敗しました: {str(e)}")
            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    # データがアップロードされている場合、データプレビューを表示
    if st.session_state.data is not None:
        st.divider()
        st.header("データプレビュー")
        
        # データの最初の行を表示
        st.subheader("データサンプル")
        st.dataframe(st.session_state.data.head(10))
        
        # データ型情報
        st.subheader("データ型情報")
        dtype_df = pd.DataFrame({
            'カラム名': st.session_state.data.columns,
            'データ型': st.session_state.data.dtypes.astype(str)
        })
        st.dataframe(dtype_df)
    else:
        st.info("👆 データファイルをアップロードしてください")
        
        st.markdown("""
        ### 使い方
        1. CSVまたはExcelファイルをアップロード
        2. カラム数が20を超える場合は、分析対象のカラムを選択
        3. 各タブで分析結果を確認：
           - **データプレビュー**: データアップロードとデータの概要確認
           - **自動分析**: 統計量、欠損値、相関、分布、外れ値の分析結果
           - **機械学習モデル構築**: モデルの選択、学習、評価
        """)
        
        st.markdown("""
        ### サポートされている機能
        - ✅ CSV、Excelファイルの読み込み
        - ✅ 基本統計量の表示
        - ✅ 欠損値の分析と可視化
        - ✅ 相関分析とヒートマップ
        - ✅ 分布の可視化
        - ✅ 外れ値の検出（IQR法）
        - ✅ 機械学習モデルの構築（回帰・分類）
        - ✅ ハイパーパラメータの調整
        - ✅ 交差検証による評価
        - ✅ 特徴量重要度の可視化
        """)
    
# 自動分析タブ
with tab2:
    if st.session_state.data is not None:
        st.header("自動分析結果")
        
        analyzer = DataAnalyzer()
        
        # 基本統計量
        st.subheader("📈 基本統計量")
        basic_stats = analyzer.get_basic_statistics(st.session_state.data)
        if not basic_stats.empty:
            st.dataframe(basic_stats)
        else:
            st.info("数値データがありません")
        
        # 欠損値分析
        st.subheader("❓ 欠損値分析")
        missing_info = analyzer.analyze_missing_values(st.session_state.data)
        if len(missing_info) > 0:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(missing_info)
            with col2:
                plot_missing_values(missing_info)
        else:
            st.success("欠損値はありません")
        
        # 相関分析
        st.subheader("🔗 相関分析")
        corr_matrix = analyzer.calculate_correlations(st.session_state.data)
        if not corr_matrix.empty:
            plot_correlation_heatmap(corr_matrix)
        else:
            st.info("相関を計算できる数値データがありません")
        
        # 分布の可視化
        st.subheader("📊 分布の可視化")
        numeric_cols = analyzer.get_numeric_columns(st.session_state.data)
        if numeric_cols:
            plot_distributions(st.session_state.data, numeric_cols)
        else:
            st.info("可視化できる数値データがありません")
        
        # 外れ値検出
        st.subheader("🎯 外れ値検出")
        outliers_iqr = analyzer.detect_outliers_iqr(st.session_state.data)
        if outliers_iqr:
            # 外れ値のサマリー
            outlier_summary = analyzer.get_outlier_summary(st.session_state.data)
            st.dataframe(outlier_summary)
            
            # 外れ値の可視化
            plot_outliers(st.session_state.data, outliers_iqr)
        else:
            st.info("外れ値は検出されませんでした")
        
        # 分布情報の詳細
        st.subheader("📋 分布情報の詳細")
        dist_info = analyzer.analyze_distributions(st.session_state.data)
        
        # 数値データの歪度と尖度
        numeric_dist_info = {k: v for k, v in dist_info.items() if v.get('type') == 'numeric'}
        if numeric_dist_info:
            dist_df = pd.DataFrame([
                {
                    'カラム名': col,
                    '歪度': info['skewness'],
                    '尖度': info['kurtosis'],
                    'ユニーク値数': info['unique_values']
                }
                for col, info in numeric_dist_info.items()
            ])
            st.dataframe(dist_df)
    else:
        st.info("データをアップロードしてから分析を行ってください")
    
# 機械学習モデル構築タブ
with tab3:
    if st.session_state.data is not None:
        st.header("機械学習モデル構築")
        
        # ターゲット変数の選択
        st.subheader("1️⃣ ターゲット変数の選択")
        target_column = st.selectbox(
            "ターゲット変数（予測したい変数）を選択してください",
            options=st.session_state.data.columns.tolist()
        )
        
        if target_column:
            # 特徴量の選択
            feature_columns = [col for col in st.session_state.data.columns if col != target_column]
            
            # 問題の種類を判定
            target_unique = st.session_state.data[target_column].nunique()
            is_numeric_target = pd.api.types.is_numeric_dtype(st.session_state.data[target_column])
            
            if is_numeric_target and target_unique > 10:
                problem_type = "regression"
                problem_type_display = "回帰問題"
            else:
                problem_type = "classification"
                problem_type_display = "分類問題"
            
            st.info(f"問題の種類: **{problem_type_display}** (ターゲット変数のユニーク値数: {target_unique})")
            
            # データの準備
            X = st.session_state.data[feature_columns]
            y = st.session_state.data[target_column]
            
            # カテゴリカル変数の処理
            categorical_cols = X.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.warning(f"カテゴリカル変数が検出されました: {', '.join(categorical_cols)}")
                encoding_method = st.selectbox(
                    "エンコーディング方法を選択",
                    ["Label Encoding", "One-Hot Encoding"]
                )
                
                if encoding_method == "Label Encoding":
                    from sklearn.preprocessing import LabelEncoder
                    for col in categorical_cols:
                        # NaN値を一時的に文字列で置換してからLabel Encoding
                        X[col] = X[col].fillna('missing')
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                else:
                    # One-Hot Encodingの場合、NaN値も適切に処理
                    X = pd.get_dummies(X, columns=categorical_cols, dummy_na=True)
            
            # 特徴量選択と次元削減
            st.subheader("1️⃣-1 特徴量選択・次元削減")
            
            # 欠損値を一時的に処理（分析用）
            X_for_analysis = X.copy()
            if X_for_analysis.isnull().any().any():
                numeric_cols_temp = X_for_analysis.select_dtypes(include=[np.number]).columns
                if len(numeric_cols_temp) > 0:
                    X_for_analysis[numeric_cols_temp] = X_for_analysis[numeric_cols_temp].fillna(X_for_analysis[numeric_cols_temp].mean())
                X_for_analysis = X_for_analysis.fillna(0)
            
            # 特徴量選択・次元削減の選択
            feature_processing_option = st.radio(
                "特徴量処理方法を選択",
                ["元の特徴量をそのまま使用", "特徴量選択のみ", "次元削減のみ", "特徴量選択 + 次元削減"],
                horizontal=True
            )
            
            if feature_processing_option != "元の特徴量をそのまま使用":
                # 統計情報の表示
                feature_selector = FeatureSelector()
                analyzer = DataAnalyzer()
                
                # 特徴量の統計情報を取得
                with st.expander("📊 特徴量統計情報", expanded=True):
                    if len(X_for_analysis.columns) > 0:
                        # 推奨特徴量の取得
                        recommendations = feature_selector.get_recommended_features(
                            X_for_analysis, y, problem_type, top_k=min(10, len(X_for_analysis.columns))
                        )
                        
                        if recommendations['recommended_features']:
                            st.success("🎯 統計的に推奨される特徴量:")
                            
                            # 推奨理由の表示
                            for rationale in recommendations['selection_rationale']:
                                st.write(f"• {rationale}")
                            
                            # ランキングテーブルの表示
                            if not recommendations['ranking_table'].empty:
                                st.subheader("特徴量ランキング")
                                ranking_display = recommendations['ranking_table'][
                                    ['feature', 'correlation_score', 'mutual_info_score', 'variance_score', 'average_rank']
                                ].round(4)
                                st.dataframe(ranking_display, use_container_width=True)
                    else:
                        st.info("数値型の特徴量がありません")
                
                # 特徴量選択
                if "特徴量選択" in feature_processing_option:
                    st.subheader("🎯 特徴量選択")
                    
                    selection_method = st.selectbox(
                        "選択方法",
                        ["推奨特徴量を使用", "手動選択", "統計ベース自動選択"]
                    )
                    
                    if selection_method == "推奨特徴量を使用":
                        if recommendations['recommended_features']:
                            selected_features = st.multiselect(
                                "特徴量を選択",
                                options=X_for_analysis.columns.tolist(),
                                default=recommendations['recommended_features'][:min(5, len(recommendations['recommended_features']))]
                            )
                        else:
                            selected_features = X_for_analysis.columns.tolist()
                    
                    elif selection_method == "手動選択":
                        selected_features = st.multiselect(
                            "特徴量を選択",
                            options=X_for_analysis.columns.tolist(),
                            default=X_for_analysis.columns.tolist()[:min(5, len(X_for_analysis.columns))]
                        )
                    
                    else:  # 統計ベース自動選択
                        col1, col2 = st.columns(2)
                        with col1:
                            selection_criteria = {
                                'correlation_threshold': st.slider(
                                    "相関閾値", 0.0, 1.0, 0.1, 0.05,
                                    help="目的変数との相関がこの値以上の特徴量を選択"
                                ),
                                'top_k': st.slider(
                                    "最大特徴量数", 1, min(20, len(X_for_analysis.columns)), 
                                    min(10, len(X_for_analysis.columns))
                                )
                            }
                        
                        with col2:
                            selection_criteria.update({
                                'mutual_info_threshold': st.slider(
                                    "相互情報量閾値", 0.0, 1.0, 0.05, 0.01,
                                    help="目的変数との相互情報量がこの値以上の特徴量を選択"
                                ),
                                'variance_threshold': st.slider(
                                    "分散閾値", 0.0, 1.0, 0.01, 0.01,
                                    help="分散がこの値以上の特徴量を選択"
                                )
                            })
                        
                        # 自動選択の実行
                        selection_result = feature_selector.interactive_feature_selection(
                            X_for_analysis, y, selection_criteria, problem_type
                        )
                        
                        selected_features = selection_result['final_features']
                        
                        # 選択プロセスの表示
                        st.info(f"選択結果: {len(selected_features)}個の特徴量が選択されました")
                        for step in selection_result['selection_steps']:
                            st.write(f"• {step['step_name']}: {step['features_before']} → {step['features_after']} 特徴量")
                    
                    # 選択された特徴量でXを更新
                    if selected_features:
                        X = X[selected_features]
                        X_for_analysis = X_for_analysis[selected_features]
                        st.success(f"✅ {len(selected_features)}個の特徴量が選択されました")
                    else:
                        st.warning("特徴量が選択されていません。元の特徴量を使用します。")
                
                # 次元削減
                if "次元削減" in feature_processing_option:
                    st.subheader("📉 次元削減")
                    
                    dim_reduction_method = st.selectbox(
                        "次元削減手法",
                        ["PCA (主成分分析)", "SVD (特異値分解)"]
                    )
                    
                    if dim_reduction_method == "PCA (主成分分析)":
                        col1, col2 = st.columns(2)
                        with col1:
                            pca_method = st.radio(
                                "PCA設定方法",
                                ["成分数を指定", "累積寄与率を指定"]
                            )
                        
                        with col2:
                            standardize_pca = st.checkbox("標準化を行う", value=True)
                        
                        if pca_method == "成分数を指定":
                            n_components = st.slider(
                                "主成分数", 1, min(10, len(X_for_analysis.columns)), 
                                min(3, len(X_for_analysis.columns))
                            )
                            pca_data, pca_info = analyzer.apply_pca(
                                X_for_analysis, n_components=n_components, standardize=standardize_pca
                            )
                        else:
                            variance_threshold = st.slider(
                                "累積寄与率", 0.5, 0.99, 0.95, 0.01
                            )
                            pca_data, pca_info = analyzer.apply_pca(
                                X_for_analysis, variance_threshold=variance_threshold, standardize=standardize_pca
                            )
                        
                        if not pca_data.empty:
                            # PCA結果の表示
                            st.success(f"✅ PCA完了: {pca_info['n_components']}個の主成分を生成")
                            
                            # 寄与率の表示
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**各主成分の寄与率**")
                                for i, ratio in enumerate(pca_info['explained_variance_ratio']):
                                    st.write(f"PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
                            
                            with col2:
                                st.write("**累積寄与率**")
                                for i, ratio in enumerate(pca_info['cumulative_variance_ratio']):
                                    st.write(f"PC1-PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
                            
                            # 主成分負荷量による特徴量重要度
                            if 'components' in pca_info and 'original_features' in pca_info:
                                importance = analyzer.get_feature_importance_from_components(
                                    pca_info['components'],
                                    pca_info['original_features'],
                                    pca_info['explained_variance_ratio']
                                )
                                
                                if importance:
                                    st.subheader("主成分負荷量による特徴量重要度")
                                    importance_df = pd.DataFrame(
                                        list(importance.items()),
                                        columns=['特徴量', '重要度']
                                    ).sort_values('重要度', ascending=False)
                                    
                                    fig = px.bar(
                                        importance_df,
                                        x='重要度',
                                        y='特徴量',
                                        orientation='h',
                                        title="主成分負荷量による特徴量重要度"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Xを主成分データで更新
                            X = pca_data
                    
                    else:  # SVD
                        col1, col2 = st.columns(2)
                        with col1:
                            n_components_svd = st.slider(
                                "成分数", 1, min(10, len(X_for_analysis.columns)), 
                                min(3, len(X_for_analysis.columns))
                            )
                        with col2:
                            standardize_svd = st.checkbox("標準化を行う", value=True, key="svd_standardize")
                        
                        svd_data, svd_info = analyzer.apply_svd(
                            X_for_analysis, n_components=n_components_svd, standardize=standardize_svd
                        )
                        
                        if not svd_data.empty:
                            # SVD結果の表示
                            st.success(f"✅ SVD完了: {svd_info['n_components']}個の成分を生成")
                            
                            # 特異値と寄与率の表示
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**各成分の寄与率**")
                                for i, ratio in enumerate(svd_info['explained_variance_ratio']):
                                    st.write(f"SVD{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
                            
                            with col2:
                                st.write("**累積寄与率**")
                                for i, ratio in enumerate(svd_info['cumulative_variance_ratio']):
                                    st.write(f"SVD1-SVD{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
                            
                            # Xを特異値分解データで更新
                            X = svd_data
            
            # 欠損値の処理
            if X.isnull().any().any():
                st.warning("欠損値が検出されました")
                
                # 欠損値処理方法の選択
                missing_method = st.selectbox(
                    "欠損値の処理方法を選択",
                    ["削除機能", "平均値で補完", "中央値で補完", "0で補完"]
                )
                
                if missing_method == "削除機能":
                    st.info("🗑️ 欠損値の多いカラムと行を削除します")
                    
                    # 削除閾値の設定
                    col1, col2 = st.columns(2)
                    with col1:
                        column_threshold = st.slider(
                            "カラム削除閾値（%）",
                            min_value=1,
                            max_value=50,
                            value=10,
                            step=1,
                            help="この値以上の欠損率を持つカラムを削除"
                        ) / 100
                    
                    with col2:
                        row_threshold = st.slider(
                            "行削除閾値（%）",
                            min_value=1,
                            max_value=50,
                            value=10,
                            step=1,
                            help="この値以上の欠損率を持つ行を削除"
                        ) / 100
                    
                    # データ削除の実行
                    analyzer = DataAnalyzer()
                    
                    # 全データ（特徴量 + ターゲット）に対して削除処理を実行
                    full_data = st.session_state.data.copy()
                    cleaned_data, cleaning_info = analyzer.clean_missing_data(
                        full_data, column_threshold, row_threshold
                    )
                    
                    # 削除情報の表示
                    if cleaning_info['columns_removed_count'] > 0 or cleaning_info['rows_removed_count'] > 0:
                        st.success(f"✅ データクリーニング完了")
                        
                        # 削除結果の表示
                        result_col1, result_col2, result_col3 = st.columns(3)
                        with result_col1:
                            st.metric(
                                "削除されたカラム数",
                                cleaning_info['columns_removed_count'],
                                delta=f"-{cleaning_info['columns_removed_count']}"
                            )
                        with result_col2:
                            st.metric(
                                "削除された行数",
                                cleaning_info['rows_removed_count'],
                                delta=f"-{cleaning_info['rows_removed_count']}"
                            )
                        with result_col3:
                            original_size = cleaning_info['original_shape'][0] * cleaning_info['original_shape'][1]
                            final_size = cleaning_info['final_shape'][0] * cleaning_info['final_shape'][1]
                            retention_rate = (final_size / original_size) * 100 if original_size > 0 else 0
                            st.metric(
                                "データ保持率",
                                f"{retention_rate:.1f}%"
                            )
                        
                        # 削除されたカラムの詳細表示
                        if cleaning_info['removed_columns']:
                            with st.expander("削除されたカラムの詳細"):
                                removed_info = []
                                for col in cleaning_info['removed_columns']:
                                    if col in full_data.columns:
                                        missing_pct = (full_data[col].isnull().sum() / len(full_data)) * 100
                                        removed_info.append({
                                            'カラム名': col,
                                            '欠損率': f"{missing_pct:.1f}%"
                                        })
                                if removed_info:
                                    st.dataframe(pd.DataFrame(removed_info), use_container_width=True)
                        
                        # 特徴量とターゲットを再分離
                        if target_column in cleaned_data.columns:
                            X = cleaned_data.drop(columns=[target_column])
                            y = cleaned_data[target_column]
                        else:
                            st.error(f"⚠️ ターゲット変数 '{target_column}' が削除されました。別のターゲット変数を選択してください。")
                            st.stop()
                    else:
                        st.info("削除対象のカラムまたは行がありませんでした")
                        X = st.session_state.data[feature_columns]
                        y = st.session_state.data[target_column]
                    
                    # 残りの欠損値があれば補完
                    if X.isnull().any().any():
                        st.warning("削除後もまだ欠損値があります。補完方法を選択してください。")
                        fallback_method = st.selectbox(
                            "補完方法を選択",
                            ["平均値で補完", "中央値で補完", "0で補完"],
                            key="fallback_missing"
                        )
                        
                        if fallback_method == "平均値で補完":
                            # 数値型カラムのみ平均値で補完、それ以外は0で補完
                            numeric_cols = X.select_dtypes(include=[np.number]).columns
                            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
                            X = X.fillna(0)  # 非数値カラムは0で補完
                        elif fallback_method == "中央値で補完":
                            # 数値型カラムのみ中央値で補完、それ以外は0で補完
                            numeric_cols = X.select_dtypes(include=[np.number]).columns
                            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                            X = X.fillna(0)  # 非数値カラムは0で補完
                        else:
                            X = X.fillna(0)
                
                elif missing_method == "平均値で補完":
                    # 数値型カラムのみ平均値で補完、それ以外は0で補完
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
                    X = X.fillna(0)  # 非数値カラムは0で補完
                elif missing_method == "中央値で補完":
                    # 数値型カラムのみ中央値で補完、それ以外は0で補完
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                    X = X.fillna(0)  # 非数値カラムは0で補完
                else:
                    X = X.fillna(0)
            
            # モデル選択
            st.subheader("2️⃣ モデルの選択")
            model_builder = ModelBuilder()
            available_models = model_builder.get_available_models(problem_type)
            
            # 複数モデル選択または単一モデル選択の選択
            model_selection_mode = st.radio(
                "モデル選択方式",
                ["単一モデル", "複数モデル比較"],
                horizontal=True
            )
            
            if model_selection_mode == "単一モデル":
                selected_models = [st.selectbox(
                    "使用するモデルを選択してください",
                    options=available_models
                )]
                models_config = {selected_models[0]: {}}
            else:
                st.info("複数のモデルを選択して性能を比較できます")
                selected_models = st.multiselect(
                    "比較するモデルを選択してください（複数選択可）",
                    options=available_models,
                    default=available_models[:3] if len(available_models) >= 3 else available_models
                )
                
                if not selected_models:
                    st.warning("少なくとも1つのモデルを選択してください")
                    st.stop()
                
                models_config = {model: {} for model in selected_models}
            
            # ハイパーパラメータ設定
            st.subheader("3️⃣ ハイパーパラメータの設定")
            
            # 各モデルのハイパーパラメータ設定
            for model_name in selected_models:
                with st.expander(f"📊 {model_name} のハイパーパラメータ調整", expanded=len(selected_models)==1):
                    default_params = model_builder.get_hyperparameters(problem_type, model_name)
                    hyperparameters = {}
                    
                    # 2カラムレイアウト
                    col1, col2 = st.columns(2)
                    param_count = 0
                    
                    for param, default_value in default_params.items():
                        # パラメータを左右に振り分け
                        current_col = col1 if param_count % 2 == 0 else col2
                        
                        with current_col:
                            if param in ['random_state', 'verbose']:
                                hyperparameters[param] = default_value
                            elif param == 'n_estimators':
                                hyperparameters[param] = st.slider(
                                    f"🌳 n_estimators (木の数)",
                                    min_value=10,
                                    max_value=500,
                                    value=default_value,
                                    step=10,
                                    key=f"{model_name}_n_estimators"
                                )
                                param_count += 1
                            elif param == 'max_depth':
                                if default_value is None:
                                    use_max_depth = st.checkbox(
                                        f"🌲 max_depth を制限する", 
                                        value=False,
                                        key=f"{model_name}_use_max_depth"
                                    )
                                    if use_max_depth:
                                        hyperparameters[param] = st.slider(
                                            f"🌲 max_depth (木の深さ)",
                                            min_value=1,
                                            max_value=20,
                                            value=5,
                                            key=f"{model_name}_max_depth"
                                        )
                                    else:
                                        hyperparameters[param] = None
                                else:
                                    hyperparameters[param] = st.slider(
                                        f"🌲 max_depth (木の深さ)",
                                        min_value=1,
                                        max_value=20,
                                        value=default_value,
                                        key=f"{model_name}_max_depth"
                                    )
                                param_count += 1
                            elif param == 'learning_rate':
                                hyperparameters[param] = st.slider(
                                    f"⚡ learning_rate (学習率)",
                                    min_value=0.01,
                                    max_value=1.0,
                                    value=default_value,
                                    step=0.01,
                                    key=f"{model_name}_learning_rate"
                                )
                                param_count += 1
                            elif param == 'C':
                                hyperparameters[param] = st.slider(
                                    f"⚙️ C (正則化パラメータ)",
                                    min_value=0.01,
                                    max_value=10.0,
                                    value=default_value,
                                    step=0.01,
                                    key=f"{model_name}_C"
                                )
                                param_count += 1
                            elif param == 'num_leaves':
                                hyperparameters[param] = st.slider(
                                    f"🍃 num_leaves (葉の数)",
                                    min_value=10,
                                    max_value=300,
                                    value=default_value,
                                    step=5,
                                    key=f"{model_name}_num_leaves"
                                )
                                param_count += 1
                            elif param == 'min_samples_split':
                                hyperparameters[param] = st.slider(
                                    f"🔀 min_samples_split",
                                    min_value=2,
                                    max_value=20,
                                    value=default_value,
                                    key=f"{model_name}_min_samples_split"
                                )
                                param_count += 1
                            elif param == 'min_samples_leaf':
                                hyperparameters[param] = st.slider(
                                    f"🌿 min_samples_leaf",
                                    min_value=1,
                                    max_value=20,
                                    value=default_value,
                                    key=f"{model_name}_min_samples_leaf"
                                )
                                param_count += 1
                            elif param in ['subsample', 'colsample_bytree']:
                                hyperparameters[param] = st.slider(
                                    f"📊 {param}",
                                    min_value=0.1,
                                    max_value=1.0,
                                    value=default_value,
                                    step=0.1,
                                    key=f"{model_name}_{param}"
                                )
                                param_count += 1
                            elif param == 'gamma':
                                if default_value == 'scale':
                                    gamma_option = st.selectbox(
                                        f"⚡ gamma",
                                        ['scale', 'auto', 'custom'],
                                        key=f"{model_name}_gamma_option"
                                    )
                                    if gamma_option == 'custom':
                                        hyperparameters[param] = st.slider(
                                            f"⚡ gamma (カスタム値)",
                                            min_value=0.001,
                                            max_value=1.0,
                                            value=0.1,
                                            step=0.001,
                                            key=f"{model_name}_gamma_custom"
                                        )
                                    else:
                                        hyperparameters[param] = gamma_option
                                else:
                                    hyperparameters[param] = default_value
                                param_count += 1
                            elif param == 'kernel':
                                hyperparameters[param] = st.selectbox(
                                    f"🔮 kernel",
                                    ['rbf', 'linear', 'poly', 'sigmoid'],
                                    index=['rbf', 'linear', 'poly', 'sigmoid'].index(default_value),
                                    key=f"{model_name}_kernel"
                                )
                                param_count += 1
                            else:
                                hyperparameters[param] = default_value
                    
                    models_config[model_name] = hyperparameters
            
            # 学習方法の設定
            st.subheader("4️⃣ 学習方法の設定")
            cv_methods = model_builder.get_cv_methods()
            selected_cv_method = st.selectbox(
                "交差検証の方法を選択",
                options=cv_methods
            )
            
            if selected_cv_method in ["K-Fold", "Stratified K-Fold", "Time Series Split"]:
                cv_folds = st.slider("分割数", min_value=2, max_value=10, value=5)
            else:
                cv_folds = None
            
            # テストデータの分割比率
            test_size = st.slider(
                "テストデータの割合",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05
            )
            
            # モデルの学習
            if st.button("🚀 モデルを学習", type="primary"):
                with st.spinner("モデルを学習中..."):
                    from sklearn.model_selection import train_test_split
                    
                    # データの分割
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # モデルの学習
                    try:
                        if model_selection_mode == "単一モデル":
                            # 単一モデルの学習
                            model, metrics = model_builder.train_model(
                                problem_type=problem_type,
                                model_name=selected_models[0],
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test,
                                hyperparameters=models_config[selected_models[0]],
                                cv_method=selected_cv_method,
                                cv_folds=cv_folds if cv_folds else 5
                            )
                            
                            results = {selected_models[0]: {'model': model, 'metrics': metrics}}
                            st.session_state.trained_model = model
                            st.session_state.model_metrics = metrics
                            
                        else:
                            # 複数モデルの学習
                            results = model_builder.train_multiple_models(
                                problem_type=problem_type,
                                models_config=models_config,
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test,
                                cv_method=selected_cv_method,
                                cv_folds=cv_folds if cv_folds else 5
                            )
                            
                            # 最良モデルをセッション状態に保存
                            if results:
                                best_model_name = model_builder.get_best_model(results, problem_type)
                                st.session_state.trained_model = results[best_model_name]['model']
                                st.session_state.model_metrics = results[best_model_name]['metrics']
                        
                        st.session_state.model_results = results
                        st.success(f"✅ {len(results)}個のモデルの学習が完了しました！")
                        
                        # 評価指標の表示
                        st.subheader("📊 モデルの評価結果")
                        
                        if model_selection_mode == "複数モデル比較" and len(results) > 1:
                            # 複数モデルの比較表示
                            st.subheader("🏆 モデル性能比較")
                            comparison_df = model_builder.compare_model_performance(results, problem_type)
                            
                            # 最良モデルをハイライト
                            best_model_name = model_builder.get_best_model(results, problem_type)
                            
                            # スタイル付きのデータフレーム表示
                            styled_df = comparison_df.copy()
                            if problem_type == "regression":
                                # R²スコアが最大の行をハイライト
                                max_r2_idx = styled_df['r2'].idxmax()
                                styled_df.loc[max_r2_idx, 'model_name'] = f"🏆 {styled_df.loc[max_r2_idx, 'model_name']}"
                            else:
                                # Accuracyが最大の行をハイライト
                                max_acc_idx = styled_df['accuracy'].idxmax()
                                styled_df.loc[max_acc_idx, 'model_name'] = f"🏆 {styled_df.loc[max_acc_idx, 'model_name']}"
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # モデル性能の視覚化
                            if problem_type == "regression":
                                fig = px.bar(
                                    comparison_df,
                                    x='model_name',
                                    y='r2',
                                    title='R²スコア比較',
                                    labels={'r2': 'R²スコア', 'model_name': 'モデル'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # 複数指標の比較（選択可能）
                                st.subheader("📊 分類指標比較（選択可能）")
                                
                                # 表示する指標を選択
                                available_metrics = ['accuracy', 'precision', 'recall', 'f1']
                                if 'auc' in comparison_df.columns:
                                    available_metrics.append('auc')
                                
                                metric_labels = {
                                    'accuracy': '精度 (Accuracy)',
                                    'precision': '適合率 (Precision)',
                                    'recall': '再現率 (Recall)',
                                    'f1': 'F1スコア',
                                    'auc': 'AUC'
                                }
                                
                                selected_metrics = st.multiselect(
                                    "表示する評価指標を選択",
                                    options=available_metrics,
                                    default=['accuracy', 'f1'],
                                    format_func=lambda x: metric_labels.get(x, x)
                                )
                                
                                if selected_metrics:
                                    fig = go.Figure()
                                    
                                    for metric in selected_metrics:
                                        if metric in comparison_df.columns:
                                            fig.add_trace(go.Scatter(
                                                x=comparison_df['model_name'],
                                                y=comparison_df[metric],
                                                mode='lines+markers',
                                                name=metric_labels.get(metric, metric),
                                                line=dict(width=3),
                                                marker=dict(size=8)
                                            ))
                                    
                                    fig.update_layout(
                                        title='選択された分類指標の比較',
                                        xaxis_title='モデル',
                                        yaxis_title='スコア',
                                        yaxis=dict(range=[0, 1.1]),
                                        height=500
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("少なくとも1つの評価指標を選択してください")
                            
                            st.info(f"🏆 最良モデル: **{best_model_name}**")
                        
                        # 個別モデルの詳細結果
                        if len(results) == 1:
                            # 単一モデルの詳細表示
                            model_name = list(results.keys())[0]
                            metrics = results[model_name]['metrics']
                            model = results[model_name]['model']
                            
                            st.subheader(f"📈 {model_name} の詳細結果")
                            
                            if problem_type == "regression":
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                                with col2:
                                    st.metric("MAE", f"{metrics['mae']:.4f}")
                                with col3:
                                    st.metric("R²スコア", f"{metrics['r2']:.4f}")
                            else:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("精度 (Accuracy)", f"{metrics['accuracy']:.4f}")
                                    st.metric("適合率 (Precision)", f"{metrics['precision']:.4f}")
                                with col2:
                                    st.metric("再現率 (Recall)", f"{metrics['recall']:.4f}")
                                    st.metric("F1スコア", f"{metrics['f1']:.4f}")
                                
                                # 拡張評価指標を計算
                                enhanced_metrics = model_builder.calculate_enhanced_metrics(
                                    model, X_test, y_test, problem_type
                                )
                                
                                # 混同行列の表示
                                if 'confusion_matrix' in enhanced_metrics:
                                    st.subheader("混同行列")
                                    cm = enhanced_metrics['confusion_matrix']
                                    fig = px.imshow(
                                        cm,
                                        labels=dict(x="予測値", y="真の値", color="カウント"),
                                        x=[str(i) for i in range(len(cm))],
                                        y=[str(i) for i in range(len(cm))],
                                        text_auto=True,
                                        color_continuous_scale='Blues'
                                    )
                                    fig.update_layout(title="混同行列")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # ROC曲線とAUCの表示（2クラス分類の場合）
                                if 'roc_curve' in enhanced_metrics:
                                    st.subheader("ROC曲線とAUC")
                                    roc_data = enhanced_metrics['roc_curve']
                                    auc_score = enhanced_metrics['roc_auc']
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=roc_data['fpr'],
                                        y=roc_data['tpr'],
                                        mode='lines',
                                        name=f'ROC曲線 (AUC = {auc_score:.3f})',
                                        line=dict(color='blue', width=2)
                                    ))
                                    
                                    # 対角線（ランダム分類）
                                    fig.add_trace(go.Scatter(
                                        x=[0, 1],
                                        y=[0, 1],
                                        mode='lines',
                                        name='ランダム分類',
                                        line=dict(color='red', width=1, dash='dash')
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"ROC曲線 (AUC = {auc_score:.3f})",
                                        xaxis_title="偽陽性率 (False Positive Rate)",
                                        yaxis_title="真陽性率 (True Positive Rate)",
                                        showlegend=True,
                                        width=600,
                                        height=500
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # 交差検証の結果
                            if 'cv_scores' in metrics:
                                st.subheader("交差検証の結果")
                                cv_df = pd.DataFrame({
                                    'Fold': [f"Fold {i+1}" for i in range(len(metrics['cv_scores']))],
                                    'Score': metrics['cv_scores']
                                })
                                fig = px.bar(
                                    cv_df,
                                    x='Fold',
                                    y='Score',
                                    title=f"交差検証スコア (平均: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f})"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # 特徴量重要度
                            feature_importance = model_builder.get_feature_importance(model, X_train.columns)
                            if feature_importance:
                                st.subheader("特徴量重要度")
                                importance_df = pd.DataFrame(
                                    list(feature_importance.items()),
                                    columns=['特徴量', '重要度']
                                ).sort_values('重要度', ascending=False)
                                
                                fig = px.bar(
                                    importance_df.head(20),
                                    x='重要度',
                                    y='特徴量',
                                    orientation='h',
                                    title="上位20個の特徴量重要度"
                                )
                                fig.update_layout(height=600)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif len(results) > 1:
                            # 複数モデルの特徴量重要度比較
                            st.subheader("🔍 特徴量重要度比較")
                            
                            # 複数モデルの特徴量重要度を取得
                            models_with_importance = model_builder.get_multiple_feature_importance(results, X_train.columns)
                            
                            if models_with_importance:
                                # モデル選択ドロップダウン
                                selected_model_for_importance = st.selectbox(
                                    "特徴量重要度を表示するモデルを選択",
                                    options=list(models_with_importance.keys()),
                                    index=0
                                )
                                
                                # 選択されたモデルの特徴量重要度を表示
                                feature_importance = models_with_importance[selected_model_for_importance]
                                importance_df = pd.DataFrame(
                                    list(feature_importance.items()),
                                    columns=['特徴量', '重要度']
                                ).sort_values('重要度', ascending=False)
                                
                                fig = px.bar(
                                    importance_df.head(15),
                                    x='重要度',
                                    y='特徴量',
                                    orientation='h',
                                    title=f"特徴量重要度 ({selected_model_for_importance})"
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 複数モデルの特徴量重要度比較表を表示
                                if len(models_with_importance) > 1:
                                    st.subheader("📊 モデル間特徴量重要度比較")
                                    comparison_df = model_builder.get_feature_importance_comparison(results, X_train.columns)
                                    
                                    if not comparison_df.empty:
                                        # 日本語カラム名に変換
                                        japanese_columns = {'feature': '特徴量'}
                                        for model_name in models_with_importance.keys():
                                            if model_name in comparison_df.columns:
                                                japanese_columns[model_name] = model_name
                                        if '平均重要度' in comparison_df.columns:
                                            japanese_columns['平均重要度'] = '平均重要度'
                                        
                                        comparison_df_display = comparison_df.rename(columns=japanese_columns)
                                        st.dataframe(comparison_df_display.head(15), use_container_width=True)
                            else:
                                st.info("選択されたモデルには特徴量重要度を取得できるものがありません")
                            
                            # 分類問題の場合、選択したモデルの混同行列とROC曲線を表示
                            if problem_type == "classification":
                                st.subheader("📈 拡張評価指標")
                                
                                # モデル選択
                                selected_model_for_metrics = st.selectbox(
                                    "拡張評価指標を表示するモデルを選択",
                                    options=list(results.keys()),
                                    index=0,
                                    key="metrics_model_selector"
                                )
                                
                                # 選択されたモデルの拡張評価指標を計算
                                selected_model = results[selected_model_for_metrics]['model']
                                enhanced_metrics = model_builder.calculate_enhanced_metrics(
                                    selected_model, X_test, y_test, problem_type
                                )
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # 混同行列の表示
                                    if 'confusion_matrix' in enhanced_metrics:
                                        st.subheader(f"混同行列 ({selected_model_for_metrics})")
                                        cm = enhanced_metrics['confusion_matrix']
                                        fig = px.imshow(
                                            cm,
                                            labels=dict(x="予測値", y="真の値", color="カウント"),
                                            x=[str(i) for i in range(len(cm))],
                                            y=[str(i) for i in range(len(cm))],
                                            text_auto=True,
                                            color_continuous_scale='Blues'
                                        )
                                        fig.update_layout(height=400)
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # ROC曲線とAUCの表示（2クラス分類の場合）
                                    if 'roc_curve' in enhanced_metrics:
                                        st.subheader(f"ROC曲線 ({selected_model_for_metrics})")
                                        roc_data = enhanced_metrics['roc_curve']
                                        auc_score = enhanced_metrics['roc_auc']
                                        
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=roc_data['fpr'],
                                            y=roc_data['tpr'],
                                            mode='lines',
                                            name=f'ROC曲線 (AUC = {auc_score:.3f})',
                                            line=dict(color='blue', width=2)
                                        ))
                                        
                                        # 対角線（ランダム分類）
                                        fig.add_trace(go.Scatter(
                                            x=[0, 1],
                                            y=[0, 1],
                                            mode='lines',
                                            name='ランダム分類',
                                            line=dict(color='red', width=1, dash='dash')
                                        ))
                                        
                                        fig.update_layout(
                                            title=f"AUC = {auc_score:.3f}",
                                            xaxis_title="偽陽性率",
                                            yaxis_title="真陽性率",
                                            showlegend=True,
                                            height=400
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"モデルの学習中にエラーが発生しました: {str(e)}")
    else:
        st.info("データをアップロードしてからモデル構築を行ってください")
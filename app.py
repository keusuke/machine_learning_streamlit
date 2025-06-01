import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os

from src.components.data_loader import DataLoader
from src.components.data_analyzer import DataAnalyzer
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


# サイドバー
with st.sidebar:
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
                    st.info(f"""
                    **ファイル情報**
                    - 行数: {file_info['rows']:,}
                    - 列数: {file_info['columns']:,}
                    - メモリ使用量: {file_info['memory_usage'] / 1024 / 1024:.2f} MB
                    """)
                    
            except Exception as e:
                st.error(f"ファイルの読み込みに失敗しました: {str(e)}")
            finally:
                # 一時ファイルを削除
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# メインコンテンツ
if st.session_state.data is not None:
    # タブ作成
    tab1, tab2 = st.tabs(["📊 データプレビュー", "🔍 自動分析"])
    
    # データプレビュータブ
    with tab1:
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
    
    # 自動分析タブ
    with tab2:
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
    # データがアップロードされていない場合
    st.info("👈 サイドバーからデータファイルをアップロードしてください")
    
    st.markdown("""
    ### 使い方
    1. サイドバーからCSVまたはExcelファイルをアップロード
    2. カラム数が20を超える場合は、分析対象のカラムを選択
    3. 各タブで分析結果を確認：
       - **データプレビュー**: データの概要とプレビュー
       - **自動分析**: 統計量、欠損値、相関、分布、外れ値の分析結果
    """)
    
    st.markdown("""
    ### サポートされている機能
    - ✅ CSV、Excelファイルの読み込み
    - ✅ 基本統計量の表示
    - ✅ 欠損値の分析と可視化
    - ✅ 相関分析とヒートマップ
    - ✅ 分布の可視化
    - ✅ 外れ値の検出（IQR法）
    """)
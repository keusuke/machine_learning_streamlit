import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from io import BytesIO

from src.components.data_loader import DataLoader
from src.utils.config import MAX_COLUMNS_BEFORE_SELECTION, SUPPORTED_FILE_TYPES


class TestDataLoader:
    @pytest.fixture
    def sample_csv_data(self):
        """サンプルCSVデータを作成"""
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'score': [85.5, 90.0, 78.5, 92.5, 88.0]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_excel_data(self):
        """サンプルExcelデータを作成"""
        data = {
            'product': ['A', 'B', 'C', 'D'],
            'price': [100, 200, 150, 300],
            'quantity': [10, 5, 8, 3]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def many_columns_data(self):
        """多数のカラムを持つデータを作成"""
        columns = [f'col_{i}' for i in range(30)]
        data = {col: np.random.rand(10) for col in columns}
        return pd.DataFrame(data)
    
    def test_load_csv_file(self, sample_csv_data):
        """CSVファイルの読み込みテスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            sample_csv_data.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            loader = DataLoader()
            df = loader.load_file(tmp_path)
            
            assert df is not None
            assert len(df) == 5
            assert list(df.columns) == ['id', 'name', 'age', 'score']
            assert df['name'].iloc[0] == 'Alice'
        finally:
            os.unlink(tmp_path)
    
    def test_load_excel_file(self, sample_excel_data):
        """Excelファイルの読み込みテスト"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            sample_excel_data.to_excel(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            loader = DataLoader()
            df = loader.load_file(tmp_path)
            
            assert df is not None
            assert len(df) == 4
            assert list(df.columns) == ['product', 'price', 'quantity']
            assert df['product'].iloc[0] == 'A'
        finally:
            os.unlink(tmp_path)
    
    def test_unsupported_file_type(self):
        """サポートされていないファイル形式のテスト"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'test data')
            tmp_path = tmp.name
        
        try:
            loader = DataLoader()
            with pytest.raises(ValueError, match="Unsupported file type"):
                loader.load_file(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_column_selection_needed(self, many_columns_data):
        """カラム選択が必要な場合のテスト"""
        loader = DataLoader()
        assert loader.needs_column_selection(many_columns_data) == True
        
        # カラム数が閾値以下の場合
        small_df = many_columns_data[many_columns_data.columns[:10]]
        assert loader.needs_column_selection(small_df) == False
    
    def test_select_columns(self, many_columns_data):
        """カラム選択機能のテスト"""
        loader = DataLoader()
        selected_cols = many_columns_data.columns[:10].tolist()
        
        df_selected = loader.select_columns(many_columns_data, selected_cols)
        
        assert len(df_selected.columns) == 10
        assert list(df_selected.columns) == selected_cols
    
    def test_get_preview(self, sample_csv_data):
        """データプレビュー機能のテスト"""
        loader = DataLoader()
        preview = loader.get_preview(sample_csv_data, n_rows=3)
        
        assert len(preview) == 3
        assert preview.equals(sample_csv_data.head(3))
    
    def test_empty_file(self):
        """空のファイルのテスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write('')
            tmp_path = tmp.name
        
        try:
            loader = DataLoader()
            with pytest.raises(ValueError, match="Empty file"):
                loader.load_file(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_file_info(self, sample_csv_data):
        """ファイル情報取得のテスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            sample_csv_data.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        
        try:
            loader = DataLoader()
            df = loader.load_file(tmp_path)
            info = loader.get_file_info(df)
            
            assert info['rows'] == 5
            assert info['columns'] == 4
            assert info['memory_usage'] > 0
            assert 'dtypes' in info
        finally:
            os.unlink(tmp_path)
import pandas as pd
import os
from typing import Optional, List, Dict, Any

from src.utils.config import MAX_COLUMNS_BEFORE_SELECTION, SUPPORTED_FILE_TYPES


class DataLoader:
    """データファイルの読み込みと処理を行うクラス"""
    
    def __init__(self):
        self.supported_extensions = SUPPORTED_FILE_TYPES
        self.max_columns_before_selection = MAX_COLUMNS_BEFORE_SELECTION
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """ファイルを読み込んでDataFrameを返す"""
        # ファイルの拡張子を取得
        file_extension = os.path.splitext(file_path)[1].lower().strip('.')
        
        # サポートされているファイル形式かチェック
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: .{file_extension}. Supported types: {', '.join(self.supported_extensions)}")
        
        # ファイルを読み込む
        try:
            if file_extension == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: .{file_extension}")
            
            # 空のファイルチェック
            if df.empty:
                raise ValueError("Empty file")
            
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError("Empty file")
        except Exception as e:
            raise ValueError(f"Error loading file: {str(e)}")
    
    def needs_column_selection(self, df: pd.DataFrame) -> bool:
        """カラム選択が必要かどうかを判定"""
        return len(df.columns) > self.max_columns_before_selection
    
    def select_columns(self, df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
        """指定されたカラムのみを含むDataFrameを返す"""
        # 存在しないカラムがないかチェック
        missing_columns = set(selected_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Columns not found in dataframe: {missing_columns}")
        
        return df[selected_columns]
    
    def get_preview(self, df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
        """データのプレビューを返す"""
        return df.head(n_rows)
    
    def get_file_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """ファイルの基本情報を返す"""
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        
        # データ型の文字列表現に変換
        dtype_summary = {}
        for dtype, count in info['dtypes'].items():
            dtype_summary[str(dtype)] = count
        info['dtypes'] = dtype_summary
        
        return info
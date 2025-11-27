import os
import pandas as pd
from django.conf import settings
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class DataReaderService:
    DATA_FILENAME = 'historical_data.parquet'
    DATA_FILE_REL_PATH = os.path.join('data', DATA_FILENAME)
    DATA_FRESHNESS_HOURS = 24

    def __init__(self):
        self.file_path = os.path.join(settings.BASE_DIR, self.DATA_FILE_REL_PATH)

    def get_market_data( self, requested_metrics: List[str], force_refresh: bool = False, years_back: int = 10) -> List[Dict]:
        """
        Retrieves market data for specified metrics.
        Args:
            requested_metrics: List of column names to fetch (e.g., ['close_btc', 'volume_eth'])
            force_refresh: Whether to force data file refresh
            years_back: Number of years back for data regeneration (default 10)
        Returns: List of dictionaries containing the data 
        """

        if force_refresh or self._is_data_old():
            self._refresh_data_file(years_back=years_back)

        columns_to_read = list(set(requested_metrics + ['open_time']))

        try:
            df = self._load_filtered_frame(columns_to_read, years_back)
            return df.to_dict(orient='records')
            
        except FileNotFoundError:
            self._refresh_data_file(years_back=years_back)
            df = self._load_filtered_frame(columns_to_read, years_back)
            return df.to_dict(orient='records')
        except Exception as e:
            raise ValueError(f"Data processing error: {str(e)}")



    def _refresh_data_file(self, years_back: int = 10):
        """Triggers generation of new parquet file."""
        from marketdata.services.HistoricalDataService import HistoricalDataService
        service = HistoricalDataService()
        service.generate_parquet_file( years_back=years_back, filename=self.DATA_FILENAME)

    
    def _load_filtered_frame(self, columns_to_read: List[str], years_back: int) -> pd.DataFrame:
        """Loads parquet, ensures sufficient data and cuts to required years_back window."""
        df = pd.read_parquet(self.file_path, columns=columns_to_read, engine="pyarrow")

        if self._is_data_insufficient(df, years_back):
            self._refresh_data_file(years_back)
            df = pd.read_parquet(self.file_path, columns=columns_to_read, engine="pyarrow")

        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years_back)
        df = df[pd.to_datetime(df["open_time"]) >= cutoff_date]
        return df
    
    
    def _is_data_old(self) -> bool:
        """Checks if data file requires refresh (older than DATA_FRESHNESS_HOURS or missing)"""
        if not os.path.exists(self.file_path):
            return True
        
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(self.file_path))
        file_age = datetime.now() - last_modified_time
        
        return file_age > timedelta(hours=self.DATA_FRESHNESS_HOURS)


    def _is_data_insufficient(self, df: pd.DataFrame, required_years_back: int) -> bool:
        """Checks if parquet data doesn't cover required years back from now."""
        if df.empty or 'open_time' not in df.columns:
            return True
        
        min_date = pd.to_datetime(df['open_time'].min())
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=required_years_back)
        
        return min_date > cutoff_date


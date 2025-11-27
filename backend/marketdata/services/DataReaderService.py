import os
import pandas as pd
from django.conf import settings
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class DataReaderService:
    """
    Service for efficient reading of historical data from Parquet files.
    """
    
    DATA_FILENAME = 'data_export.parquet'
    DATA_FILE_REL_PATH = os.path.join('data', DATA_FILENAME)
    DATA_FRESHNESS_HOURS = 24

    def __init__(self):
        self.file_path = os.path.join(settings.BASE_DIR, self.DATA_FILE_REL_PATH)

    def get_market_data( self, requested_metrics: List[str], force_refresh: bool = False ) -> List[Dict]:
        """
        Retrieves market data for specified metrics.
        Args:
            requested_metrics: List of column names to fetch (e.g., ['close_btc', 'volume_eth'])
            force_refresh: Whether to force data file refresh

        Returns: List of dictionaries containing the data 
        """

        if force_refresh or self._is_data_stale():
            self._refresh_data_file()

        columns_to_read = list(set(requested_metrics + ['open_time']))

        try:
            df = pd.read_parquet(
                self.file_path, 
                columns=columns_to_read, 
                engine='pyarrow'
            )
            return df.to_dict(orient='records')
            
        except FileNotFoundError:
            raise RuntimeError("Critical error: Data file not available.")
        except Exception as e:
            raise ValueError(f"Data processing error: {str(e)}")


    def _is_data_stale(self) -> bool:
        """Checks if data file requires refresh (older than DATA_FRESHNESS_HOURS)"""
        if not os.path.exists(self.file_path):
            return True
        
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(self.file_path))
        file_age = datetime.now() - last_modified_time
        
        return file_age > timedelta(hours=self.DATA_FRESHNESS_HOURS)

    def _refresh_data_file(self):
        """Triggers generation of new Parquet file."""
        from marketdata.services.HistoricalDataService import HistoricalDataService
        
        service = HistoricalDataService()
        service.generate_parquet_file( years_back=10, filename=self.DATA_FILENAME)

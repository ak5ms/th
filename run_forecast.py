"""Write two-year batched Lasso forecasts: python run_forecast.py."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from takehome.forecast import write_lasso_forecasts

if __name__ == '__main__':
    result = write_lasso_forecasts(
        Path(os.environ.get('DATA_PATH', ROOT/'ESc1_signal_components_5min (6).parquet')),
        Path(os.environ.get('FORECAST_OUTPUT_DIR', ROOT/'forecasts')))
    print(json.dumps(result['metadata'], indent=2))
    for key in ('validation_csv', 'validation_parquet', 'oos_csv', 'oos_parquet'):
        print(f'{key}: {result[key].resolve()}')

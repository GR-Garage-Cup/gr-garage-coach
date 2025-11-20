import pandas as pd
import numpy as np
import logging
from typing import Tuple
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemetryFormatConverter:

    COLUMN_MAPPINGS = {
        'vcar': 'speed',
        'Speed': 'speed',
        'pbrake_f': 'brake_front',
        'pbrake_r': 'brake_rear',
        'lat': 'latitude',
        'lon': 'longitude',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'laptrigger_lapdist_dls': 'lap_distance',
        'Steering_Angle': 'steering_angle',
        'nmot': 'engine_rpm',
        'ath': 'throttle',
        'aps': 'throttle_position',
        'Gear': 'gear',
        'accx_can': 'accel_x',
        'accy_can': 'accel_y'
    }

    def detect_format(self, df: pd.DataFrame) -> str:
        columns_lower = [col.lower() for col in df.columns]

        if 'variable' in columns_lower and 'value' in columns_lower:
            logger.info("Detected LONG-FORMAT TRD file (variable/value pairs)")
            return 'long'

        telemetry_columns = ['speed', 'brake', 'throttle', 'latitude', 'longitude',
                            'vcar', 'pbrake', 'lat', 'lon', 'ath', 'aps']

        matching_cols = sum(1 for col in columns_lower if any(tc in col for tc in telemetry_columns))

        if matching_cols >= 3:
            logger.info("Detected WIDE-FORMAT file (ready to analyze)")
            return 'wide'

        logger.warning(f"Format unclear. Columns: {df.columns.tolist()}. Assuming wide-format.")
        return 'wide'

    def convert_long_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Converting long-format to wide-format...")
        logger.info(f"Input shape: {df.shape}")

        columns_map = {col.lower(): col for col in df.columns}

        timestamp_col = columns_map.get('timestamp', columns_map.get('time'))
        variable_col = columns_map.get('variable', columns_map.get('channel'))
        value_col = columns_map.get('value', columns_map.get('data'))

        if not all([timestamp_col, variable_col, value_col]):
            raise ValueError(
                f"Cannot find required columns. Found: {df.columns.tolist()}. "
                f"Need: timestamp/time, variable/channel, value/data"
            )

        lap_col = columns_map.get('lap', columns_map.get('lap_number'))

        logger.info(f"Pivoting {len(df)} rows...")

        if lap_col:
            wide_df = df.pivot_table(
                index=[timestamp_col, lap_col],
                columns=variable_col,
                values=value_col,
                aggfunc='first'
            ).reset_index()
        else:
            wide_df = df.pivot_table(
                index=timestamp_col,
                columns=variable_col,
                values=value_col,
                aggfunc='first'
            ).reset_index()

        logger.info(f"Pivoted shape: {wide_df.shape}")
        logger.info(f"Columns after pivot: {wide_df.columns.tolist()}")

        if isinstance(wide_df.columns, pd.MultiIndex):
            wide_df.columns = ['_'.join(col).strip('_') for col in wide_df.columns]

        wide_df = self.normalize_column_names(wide_df)

        logger.info(f"Final shape: {wide_df.shape}")
        logger.info(f"Final columns: {wide_df.columns.tolist()}")

        return wide_df

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        rename_dict = {}
        for col in df.columns:
            for old_name, new_name in self.COLUMN_MAPPINGS.items():
                if old_name.lower() in col.lower():
                    rename_dict[col] = new_name
                    break

        if rename_dict:
            logger.info(f"Renaming columns: {rename_dict}")
            df = df.rename(columns=rename_dict)

        return df

    def convert_to_wide_format(self, file_content: bytes) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            logger.info(f"Loaded CSV with {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise ValueError(f"Cannot read CSV file: {e}")

        format_type = self.detect_format(df)

        if format_type == 'long':
            logger.info("Converting from long-format to wide-format...")
            df = self.convert_long_to_wide(df)
        else:
            logger.info("File is already in wide-format, normalizing column names...")
            df = self.normalize_column_names(df)

        if len(df) == 0:
            raise ValueError("Converted DataFrame is empty")

        logger.info(f"Final DataFrame: {len(df)} rows, columns: {df.columns.tolist()}")

        return df


def main():
    converter = TelemetryFormatConverter()

    wide_file = "/Users/ldm/Desktop/GARAGE COACH/backend/sample_road_america_lap.csv"
    with open(wide_file, 'rb') as f:
        content = f.read()
        df = converter.convert_to_wide_format(content)
        print(f"\nWide-format test: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:10]}")
        print(df.head(2))


if __name__ == "__main__":
    main()

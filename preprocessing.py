"""
Data Preprocessing Module
==========================
Handles data loading, cleaning, and preprocessing for reservoir production optimization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocess reservoir production data"""
    
    def __init__(self, scaling_method='standard'):
        """
        Initialize preprocessor
        
        Args:
            scaling_method: 'standard', 'robust', or 'minmax'
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
        # Initialize scaler
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
    
    def load_data(self, filepath):
        """Load data from CSV"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values
        
        Args:
            df: Input dataframe
            strategy: 'mean', 'median', 'knn'
        """
        print("\nHandling missing values...")
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            print("No missing values found!")
            return df
        
        # Separate numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Handle numeric columns
        if strategy in ['mean', 'median']:
            self.imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        elif strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # Handle categorical columns (mode)
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        missing_after = df.isnull().sum().sum()
        print(f"Missing values: {missing_before} → {missing_after}")
        
        return df
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=3):
        """
        Remove outliers from numeric columns
        
        Args:
            df: Input dataframe
            columns: List of columns to check (None = all numeric)
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or z-score threshold
        """
        print(f"\nRemoving outliers using {method} method...")
        original_len = len(df)
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
                df = df[z_scores < threshold]
        
        removed = original_len - len(df)
        print(f"Removed {removed} outliers ({removed/original_len*100:.2f}%)")
        
        return df.reset_index(drop=True)
    
    def create_time_features(self, df, date_column='date'):
        """Extract time-based features"""
        print("\nCreating time features...")
        
        df[date_column] = pd.to_datetime(df[date_column])
        
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['day_of_year'] = df[date_column].dt.dayofyear
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        print("Added: year, month, quarter, day_of_week, day_of_year, is_weekend, month_sin, month_cos")
        
        return df
    
    def create_lag_features(self, df, columns, lags=[1, 7, 30], groupby='well_id'):
        """Create lag features for time series"""
        print(f"\nCreating lag features for {columns}...")
        
        df = df.sort_values([groupby, 'date'])
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby(groupby)[col].shift(lag)
        
        print(f"Added {len(columns) * len(lags)} lag features")
        
        return df
    
    def create_rolling_features(self, df, columns, windows=[7, 30], groupby='well_id'):
        """Create rolling window statistics"""
        print(f"\nCreating rolling features for {columns}...")
        
        df = df.sort_values([groupby, 'date'])
        
        for col in columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df.groupby(groupby)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_{window}'] = df.groupby(groupby)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
        
        print(f"Added {len(columns) * len(windows) * 2} rolling features")
        
        return df
    
    def encode_categorical(self, df, method='onehot'):
        """Encode categorical variables"""
        print(f"\nEncoding categorical variables using {method}...")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['date', 'well_id']]
        
        if len(categorical_cols) == 0:
            print("No categorical columns to encode")
            return df
        
        if method == 'onehot':
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in categorical_cols:
                df[f'{col}_encoded'] = le.fit_transform(df[col])
        
        print(f"Encoded {len(categorical_cols)} categorical columns")
        
        return df
    
    def scale_features(self, df, columns=None, fit=True):
        """Scale numeric features"""
        print(f"\nScaling features using {self.scaling_method} scaler...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            # Exclude certain columns
            exclude = ['well_id', 'year', 'month', 'quarter', 'day_of_week']
            columns = [col for col in columns if col not in exclude]
        
        if fit:
            df[columns] = self.scaler.fit_transform(df[columns])
        else:
            df[columns] = self.scaler.transform(df[columns])
        
        print(f"Scaled {len(columns)} features")
        
        return df
    
    def prepare_features_target(self, df, target_column, exclude_columns=None):
        """Prepare features and target for ML"""
        print(f"\nPreparing features and target...")
        print(f"Target variable: {target_column}")
        
        if exclude_columns is None:
            exclude_columns = ['date', 'well_id', 'status']
        
        # Separate features and target
        y = df[target_column].copy()
        
        # Remove target and excluded columns from features
        X = df.drop(columns=[target_column] + exclude_columns, errors='ignore')
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Samples: {len(X):,}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        return X, y
    
    def get_feature_info(self, df):
        """Get comprehensive feature information"""
        info = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        return info


def preprocess_pipeline(filepath, target='oil_rate', test_size=0.2):
    """
    Complete preprocessing pipeline
    
    Args:
        filepath: Path to data file
        target: Target variable name
        test_size: Train-test split ratio
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    from sklearn.model_selection import train_test_split
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaling_method='robust')
    
    # Load data
    df = preprocessor.load_data(filepath)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, strategy='mean')
    
    # Create time features
    if 'date' in df.columns:
        df = preprocessor.create_time_features(df, date_column='date')
    
    # Create lag features for production rates
    production_cols = ['oil_rate', 'gas_rate', 'water_rate', 'reservoir_pressure']
    production_cols = [col for col in production_cols if col in df.columns]
    
    if 'well_id' in df.columns:
        df = preprocessor.create_lag_features(df, production_cols, lags=[1, 7, 30])
        df = preprocessor.create_rolling_features(df, production_cols, windows=[7, 30])
    
    # Remove rows with NaN from lag features
    df = df.dropna()
    
    # Encode categorical variables
    df = preprocessor.encode_categorical(df, method='onehot')
    
    # Prepare features and target
    X, y = preprocessor.prepare_features_target(df, target_column=target)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    
    # Scale features
    X_train = preprocessor.scale_features(X_train, fit=True)
    X_test = preprocessor.scale_features(X_test, fit=False)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")
    print(f"Number of features: {X_train.shape[1]}")
    print("="*60)
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        'data/synthetic/full_dataset.csv',
        target='oil_rate'
    )
    
    print("\nFeature names:")
    print(preprocessor.feature_names[:10], "...")

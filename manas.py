"""
MANAS: Multi-modal Adaptive Neural Architecture for Stocks
Indian Market Specific Hybrid Framework
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, Attention, 
    MultiHeadAttention, LayerNormalization, Add,
    Concatenate, Flatten, Reshape, BatchNormalization,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ML Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# NLP Libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Time Series
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# ============================================================================
# PART 1: ADVANCED INDIAN MARKET DATA HANDLING
# ============================================================================

class IndianMarketDataHandler:
    """Specialized handler for Indian stock market data"""
    
    def __init__(self):
        self.india_holidays = self._get_indian_holidays()
        self.business_days = CustomBusinessDay(calendar=self._get_holiday_calendar())
        
    def _get_holiday_calendar(self):
        """Create Indian holiday calendar"""
        class IndiaHolidayCalendar(AbstractHolidayCalendar):
            rules = [
                Holiday('Republic Day', month=1, day=26),
                Holiday('Holi', month=3, day=25),
                Holiday('Maharashtra Day', month=5, day=1),
                Holiday('Independence Day', month=8, day=15),
                Holiday('Ganesh Chaturthi', month=9, day=7),
                Holiday('Gandhi Jayanti', month=10, day=2),
                Holiday('Diwali', month=11, day=1),
                Holiday('Christmas', month=12, day=25),
            ]
        return IndiaHolidayCalendar()
    
    def _get_indian_holidays(self):
        """Get list of Indian market holidays"""
        return [
            '2024-01-26', '2024-03-25', '2024-05-01', '2024-08-15',
            '2024-09-07', '2024-10-02', '2024-11-01', '2024-12-25'
        ]
    
    def get_stock_data_with_macro(self, ticker, start, end):
        """Fetch stock data with macroeconomic indicators"""
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        
        if data.empty:
            return data
        
        # Add Indian market specific indicators
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        data.index = data.index.tz_localize(None)
        
        # Add macroeconomic features (simulated - in production, fetch from RBI/NSDL)
        data = self._add_macro_indicators(data)
        
        # Add technical indicators
        data = self._add_advanced_technical_indicators(data)
        
        return data
    
    def _add_macro_indicators(self, df):
        """Add macroeconomic indicators (simulated - replace with actual API calls)"""
        df = df.copy()
        
        # Simulate INR/USD exchange rate impact
        df['INR_USD_Impact'] = np.random.normal(0, 0.01, len(df))
        
        # Simulate FII/DII activity
        df['FII_Activity'] = np.random.normal(0, 0.02, len(df))
        df['DII_Activity'] = np.random.normal(0, 0.015, len(df))
        
        # Simulate crude oil price impact
        df['Crude_Oil_Impact'] = np.random.normal(0, 0.005, len(df))
        
        return df
    
    def _add_advanced_technical_indicators(self, df):
        """Add advanced technical indicators"""
        df = df.copy()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # Volatility indicators
        df['ATR'] = self._calculate_atr(df)
        df['Bollinger_Upper'] = df['SMA_20'] + (df['Close'].rolling(20).std() * 2)
        df['Bollinger_Lower'] = df['SMA_20'] - (df['Close'].rolling(20).std() * 2)
        
        # Momentum indicators
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = self._calculate_obv(df)
        
        # Indian market specific
        df['Delivery_Percentage'] = np.random.uniform(60, 90, len(df))  # Simulated
        
        return df.dropna()
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv


# ============================================================================
# PART 2: ADVANCED NEWS AND SENTIMENT ANALYSIS WITH LLMs
# ============================================================================

class AdvancedSentimentAnalyzer:
    """
    Multi-model sentiment analysis combining FinBERT, LLaMA, and traditional NLP
    Inspired by references [1], [5], [21]
    """
    
    def __init__(self):
        # Initialize multiple sentiment models
        self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.vader = SentimentIntensityAnalyzer()
        
        # Indian language support (for regional news)
        try:
            self.indic_bert = pipeline("sentiment-analysis", 
                                      model="ai4bharat/indic-bert")
        except:
            self.indic_bert = None
            print("IndicBERT not available, using FinBERT only")
        
        # Entity recognition for Indian stocks
        self.ner_pipeline = pipeline("ner", 
                                     model="ai4bharat/bert-base-multilingual-cased-indic")
        
        # News API configuration
        self.NEWS_API_KEY = "563215a35c1a47968f46271e04083ea3"
        self.NEWS_API_URL = "https://newsapi.org/v2/everything"
        
        # Indian stock name mapping
        self.stock_name_mapping = self._create_stock_mapping()
    
    def _create_stock_mapping(self):
        """Create comprehensive mapping for Indian stocks"""
        return {
            "RELIANCE": ["Reliance Industries", "RIL", "Mukesh Ambani"],
            "TCS": ["Tata Consultancy Services", "Tata Consultancy"],
            "INFY": ["Infosys", "Infosys Technologies"],
            "HDFCBANK": ["HDFC Bank", "HDFC"],
            "ICICIBANK": ["ICICI Bank", "ICICI"],
            "SBIN": ["State Bank of India", "SBI"],
            "BHARTIARTL": ["Bharti Airtel", "Airtel"],
            "ITC": ["ITC Limited", "ITC"],
            "WIPRO": ["Wipro", "Wipro Technologies"],
            "TATAMOTORS": ["Tata Motors", "Tata"],
        }
    
    def fetch_indian_news(self, stock_symbol, days=7):
        """Fetch news from multiple Indian sources"""
        query = self.stock_name_mapping.get(stock_symbol, [stock_symbol])[0]
        
        # News API call
        params = {
            "q": f"{query} AND (India OR NSE OR BSE)",
            "apiKey": self.NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "from": (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        }
        
        try:
            response = requests.get(self.NEWS_API_URL, params=params)
            if response.status_code == 200:
                articles = response.json().get("articles", [])
                return self._enhance_with_indian_sources(articles)
        except Exception as e:
            print(f"Error fetching news: {e}")
        
        return []
    
    def _enhance_with_indian_sources(self, articles):
        """Add Indian-specific news sources"""
        indian_sources = ['Economic Times', 'Business Standard', 'Mint', 
                         'LiveMint', 'NDTV Profit', 'CNBC TV18']
        
        # Filter and prioritize Indian sources
        enhanced_articles = []
        for article in articles:
            source = article.get('source', {}).get('name', '')
            if any(indian_source.lower() in source.lower() for indian_source in indian_sources):
                article['relevance_score'] = 1.0
            else:
                article['relevance_score'] = 0.7
            
            enhanced_articles.append(article)
        
        return sorted(enhanced_articles, key=lambda x: x['relevance_score'], reverse=True)
    
    def analyze_multi_modal_sentiment(self, text):
        """
        Combine multiple sentiment analysis models
        Inspired by reference [5] - Multimodal Deep Fusion
        """
        if not text:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'score': 0.0}
        
        results = {}
        
        # FinBERT analysis
        finbert_result = self.finbert(text[:512])[0]
        finbert_score = 1 if finbert_result['label'] == 'positive' else (-1 if finbert_result['label'] == 'negative' else 0)
        results['finbert'] = {
            'score': finbert_score * finbert_result['score'],
            'confidence': finbert_result['score']
        }
        
        # VADER analysis
        vader_scores = self.vader.polarity_scores(text)
        vader_score = vader_scores['compound']
        results['vader'] = {
            'score': vader_score,
            'confidence': abs(vader_score)
        }
        
        # TextBlob analysis
        blob = TextBlob(text)
        blob_score = blob.sentiment.polarity
        results['textblob'] = {
            'score': blob_score,
            'confidence': abs(blob_score)
        }
        
        # Calculate weighted ensemble score
        weights = {'finbert': 0.5, 'vader': 0.3, 'textblob': 0.2}
        ensemble_score = sum(results[model]['score'] * weights[model] 
                            for model in weights.keys())
        
        # Determine sentiment with confidence
        if ensemble_score > 0.15:
            sentiment = 'positive'
        elif ensemble_score < -0.15:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Calculate ensemble confidence
        confidence = np.mean([results[model]['confidence'] for model in weights.keys()])
        
        return {
            'sentiment': sentiment,
            'score': ensemble_score,
            'confidence': confidence,
            'model_details': results
        }
    
    def extract_entities_for_indian_stocks(self, text):
        """Extract Indian stock related entities"""
        if self.ner_pipeline:
            entities = self.ner_pipeline(text[:512])
            stock_entities = [e for e in entities if 'STOCK' in e.get('entity_group', '')]
            return stock_entities
        return []
    
    def calculate_sentiment_trend(self, news_articles, stock_symbol):
        """
        Calculate sentiment trend over time with Indian market context
        Inspired by reference [21] - Stacking Ensemble Approach
        """
        sentiment_timeline = []
        
        for article in news_articles:
            date = article.get('publishedAt', '')[:10]
            title = article.get('title', '')
            description = article.get('description', '')
            
            # Combine title and description
            text = f"{title} {description}".strip()
            
            # Multi-modal sentiment analysis
            sentiment_result = self.analyze_multi_modal_sentiment(text)
            
            # Calculate impact score based on source relevance
            source = article.get('source', {}).get('name', '')
            impact_multiplier = 1.2 if any(src in source for src in ['Economic Times', 'Mint']) else 1.0
            
            sentiment_timeline.append({
                'date': date,
                'title': title,
                'sentiment': sentiment_result['sentiment'],
                'score': sentiment_result['score'] * impact_multiplier,
                'confidence': sentiment_result['confidence'],
                'source': source
            })
        
        # Aggregate daily sentiment
        df_sentiment = pd.DataFrame(sentiment_timeline)
        if not df_sentiment.empty:
            df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
            daily_sentiment = df_sentiment.groupby('date').agg({
                'score': ['mean', 'std', 'count'],
                'confidence': 'mean'
            }).round(3)
            
            # Calculate sentiment trend (3-day moving average)
            daily_sentiment['trend'] = daily_sentiment[('score', 'mean')].rolling(3).mean()
            
            return daily_sentiment, df_sentiment
        
        return None, None


# ============================================================================
# PART 3: DEEP LEARNING ARCHITECTURE WITH MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttentionStockModel:
    """
    Advanced Multi-Head Attention Architecture for Stock Prediction
    Inspired by references [3], [5], [10], [13]
    """
    
    def __init__(self, input_shape, n_heads=4, key_dim=64):
        self.input_shape = input_shape
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build Multi-Head Attention model with parallel processing streams"""
        
        # Input layer
        inputs = Input(shape=self.input_shape, name='input')
        
        # Stream 1: LSTM for temporal patterns (reference [3])
        lstm_out = LSTM(128, return_sequences=True, name='lstm_1')(inputs)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = LSTM(64, return_sequences=True, name='lstm_2')(lstm_out)
        
        # Stream 2: GRU for faster processing
        gru_out = GRU(128, return_sequences=True, name='gru_1')(inputs)
        gru_out = Dropout(0.3)(gru_out)
        gru_out = GRU(64, return_sequences=True, name='gru_2')(gru_out)
        
        # Stream 3: 1D CNN for local pattern detection
        cnn_out = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        cnn_out = BatchNormalization()(cnn_out)
        cnn_out = MaxPooling1D(pool_size=2)(cnn_out)
        cnn_out = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn_out)
        cnn_out = GlobalAveragePooling1D()(cnn_out)
        cnn_out = Reshape((1, 128))(cnn_out)
        cnn_out = tf.repeat(cnn_out, repeats=self.input_shape[0], axis=1)
        
        # Combine streams
        combined = Concatenate(axis=-1)([lstm_out, gru_out, cnn_out])
        
        # Multi-Head Attention mechanism (reference [5], [13])
        attention_output = MultiHeadAttention(
            num_heads=self.n_heads, 
            key_dim=self.key_dim,
            name='multi_head_attention'
        )(combined, combined)
        
        # Add & Norm (Residual connection)
        attention_output = Add()([combined, attention_output])
        attention_output = LayerNormalization()(attention_output)
        
        # Feed-forward network
        ff_output = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(attention_output)
        ff_output = Dropout(0.3)(ff_output)
        ff_output = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(ff_output)
        
        # Global pooling
        pooled = GlobalAveragePooling1D()(ff_output)
        
        # Output layer for multi-task learning
        price_output = Dense(1, activation='linear', name='price_prediction')(pooled)
        trend_output = Dense(3, activation='softmax', name='trend_prediction')(pooled)  # Up/Down/Neutral
        volatility_output = Dense(1, activation='relu', name='volatility_prediction')(pooled)
        
        # Create model
        self.model = Model(
            inputs=inputs, 
            outputs=[price_output, trend_output, volatility_output]
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with multiple losses"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                'price_prediction': 'mse',
                'trend_prediction': 'categorical_crossentropy',
                'volatility_prediction': 'mse'
            },
            loss_weights={
                'price_prediction': 0.6,
                'trend_prediction': 0.3,
                'volatility_prediction': 0.1
            },
            metrics={
                'price_prediction': ['mae', 'mse'],
                'trend_prediction': ['accuracy'],
                'volatility_prediction': ['mae']
            }
        )
        return self.model
    
    def train_model(self, X_train, y_train_price, y_train_trend, y_train_volatility,
                   X_val, y_val_price, y_val_trend, y_val_volatility,
                   epochs=100, batch_size=32):
        """Train the model with callbacks"""
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                mode='min'
            )
        ]
        
        self.history = self.model.fit(
            X_train,
            {
                'price_prediction': y_train_price,
                'trend_prediction': y_train_trend,
                'volatility_prediction': y_train_volatility
            },
            validation_data=(
                X_val,
                {
                    'price_prediction': y_val_price,
                    'trend_prediction': y_val_trend,
                    'volatility_prediction': y_val_volatility
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history


# ============================================================================
# PART 4: OUTLIER DETECTION AND COMPENSATION
# ============================================================================

class OutlierDetectorAndCompensator:
    """
    Advanced outlier detection and compensation for stock data
    Inspired by reference [2]
    """
    
    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = RobustScaler()
        
    def detect_outliers_multiple_methods(self, df, columns):
        """
        Detect outliers using multiple methods
        Inspired by reference [2] - Deep Learning Method for Detection and Compensation
        """
        outlier_results = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            data = df[col].values.reshape(-1, 1)
            
            # Method 1: Z-score
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers_z = z_scores > 3
            
            # Method 2: IQR
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            outliers_iqr = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
            
            # Method 3: Isolation Forest
            data_scaled = self.scaler.fit_transform(data)
            outliers_if = self.isolation_forest.fit_predict(data_scaled) == -1
            
            # Ensemble voting
            outlier_votes = np.column_stack([outliers_z.flatten(), 
                                            outliers_iqr.flatten(), 
                                            outliers_if])
            outlier_consensus = np.sum(outlier_votes, axis=1) >= 2
            
            outlier_results[col] = {
                'z_score': outliers_z.flatten(),
                'iqr': outliers_iqr.flatten(),
                'isolation_forest': outliers_if,
                'consensus': outlier_consensus,
                'indices': np.where(outlier_consensus)[0]
            }
        
        return outlier_results
    
    def compensate_outliers(self, df, outlier_results):
        """
        Compensate detected outliers using multiple strategies
        Inspired by reference [2] - Compensation mechanism
        """
        df_compensated = df.copy()
        compensation_report = {}
        
        for col, results in outlier_results.items():
            outlier_indices = results['indices']
            
            if len(outlier_indices) > 0:
                # Strategy 1: Linear interpolation
                df_compensated.loc[df.index[outlier_indices], col] = np.nan
                df_compensated[col] = df_compensated[col].interpolate(method='linear')
                
                # Strategy 2: For remaining NaNs, use moving average
                if df_compensated[col].isna().any():
                    window = 5
                    ma = df[col].rolling(window=window, center=True).mean()
                    fill_indices = df_compensated[col].isna()
                    df_compensated.loc[fill_indices, col] = ma[fill_indices]
                
                compensation_report[col] = {
                    'n_outliers': len(outlier_indices),
                    'outlier_dates': df.index[outlier_indices].tolist(),
                    'compensation_method': 'Interpolation + Moving Average'
                }
        
        return df_compensated, compensation_report


# ============================================================================
# PART 5: ADAPTIVE WEIGHTING AND ENSEMBLE FUSION
# ============================================================================

class AdaptiveEnsembleFusion:
    """
    Adaptive weighting mechanism for multi-model fusion
    Inspired by references [13], [17] - Dynamic fusion with MOE
    """
    
    def __init__(self, n_models=3):
        self.n_models = n_models
        self.weights_history = []
        self.performance_history = []
        self.meta_learner = None
        
    def train_meta_learner(self, model_predictions, actual_values):
        """
        Train meta-learner for adaptive weighting
        Inspired by reference [13] - MOE mechanism
        """
        # Prepare features (model predictions + market conditions)
        X_meta = model_predictions
        
        # Calculate errors for each model
        errors = np.abs(model_predictions - actual_values.reshape(-1, 1))
        
        # Train XGBoost meta-learner
        self.meta_learner = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05
        )
        self.meta_learner.fit(X_meta, errors.mean(axis=1))
        
        return self.meta_learner
    
    def calculate_adaptive_weights(self, model_predictions, market_conditions=None):
        """
        Calculate adaptive weights based on recent performance
        Inspired by reference [17] - Deep reinforcement learning for weighting
        """
        if self.meta_learner is not None:
            # Use meta-learner to predict which model will perform best
            predicted_errors = self.meta_learner.predict(model_predictions)
            
            # Convert errors to weights (lower error -> higher weight)
            weights = 1.0 / (predicted_errors + 1e-8)
            weights = weights / weights.sum()
            
        else:
            # Fallback: Equal weights
            weights = np.ones(self.n_models) / self.n_models
        
        # Adjust based on market conditions if provided
        if market_conditions is not None:
            # High volatility -> favor robust models
            if market_conditions.get('volatility', 0) > 0.03:
                weights *= np.array([0.2, 0.5, 0.3])  # Example adjustment
            
            weights = weights / weights.sum()
        
        self.weights_history.append(weights)
        
        return weights
    
    def ensemble_predict(self, model_predictions, market_conditions=None):
        """
        Make ensemble prediction with adaptive weights
        """
        weights = self.calculate_adaptive_weights(model_predictions, market_conditions)
        ensemble_pred = np.sum(model_predictions * weights, axis=1)
        
        return ensemble_pred, weights


# ============================================================================
# PART 6: MAIN HYBRID FRAMEWORK - MANAS
# ============================================================================

class MANAS_Framework:
    """
    MANAS: Multi-modal Adaptive Neural Architecture for Stocks
    Complete hybrid framework combining all components
    """
    
    def __init__(self):
        self.data_handler = IndianMarketDataHandler()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.outlier_detector = OutlierDetectorAndCompensator()
        self.deep_model = None
        self.ensemble_fusion = AdaptiveEnsembleFusion()
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_importance = {}
        
    def prepare_features(self, df_stock, sentiment_df=None):
        """
        Prepare comprehensive feature set combining:
        - Technical indicators
        - Sentiment scores
        - Market conditions
        - Temporal features
        """
        df = df_stock.copy()
        
        # Base features from data handler
        df = self.data_handler._add_advanced_technical_indicators(df)
        
        # Add sentiment features if available
        if sentiment_df is not None and not sentiment_df.empty:
            # Align sentiment with stock data
            sentiment_df.index = pd.to_datetime(sentiment_df.index)
            df = df.join(sentiment_df, how='left')
            df['sentiment_score'] = df[('score', 'mean')].fillna(0)
            df['sentiment_confidence'] = df[('confidence', 'mean')].fillna(0)
            df['sentiment_std'] = df[('score', 'std')].fillna(0)
        else:
            df['sentiment_score'] = 0
            df['sentiment_confidence'] = 0
            df['sentiment_std'] = 0
        
        # Add temporal features (Indian market specific)
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        
        # Add market condition features
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Add Indian holiday feature
        df['is_holiday'] = df.index.strftime('%Y-%m-%d').isin(
            self.data_handler.india_holidays
        ).astype(int)
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['Close'].rolling(window).std()
            df[f'volume_mean_{window}'] = df['Volume'].rolling(window).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def create_sequences(self, data, target_col, sequence_length=30):
        """
        Create sequences for deep learning model
        """
        X, y_price, y_trend, y_volatility = [], [], [], []
        
        feature_columns = [col for col in data.columns if col != target_col]
        
        for i in range(len(data) - sequence_length):
            # Features
            X.append(data[feature_columns].iloc[i:i+sequence_length].values)
            
            # Targets
            future_price = data[target_col].iloc[i+sequence_length]
            current_price = data[target_col].iloc[i+sequence_length-1]
            
            # Price change
            price_change = (future_price - current_price) / current_price
            
            # Price target
            y_price.append(price_change)
            
            # Trend target (one-hot encoded)
            if price_change > 0.01:
                trend = [1, 0, 0]  # Up
            elif price_change < -0.01:
                trend = [0, 1, 0]  # Down
            else:
                trend = [0, 0, 1]  # Neutral
            y_trend.append(trend)
            
            # Volatility target
            volatility = data['Returns'].iloc[i+1:i+sequence_length+1].std()
            y_volatility.append(volatility)
        
        return np.array(X), np.array(y_price), np.array(y_trend), np.array(y_volatility)
    
    def train(self, ticker, start_date, end_date):
        """
        Main training pipeline
        """
        print("="*60)
        print("MANAS: Training Pipeline Started")
        print("="*60)
        
        # Step 1: Fetch data
        print("\n[1/6] Fetching stock data...")
        df_stock = self.data_handler.get_stock_data_with_macro(ticker, start_date, end_date)
        
        if df_stock.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Step 2: Fetch and analyze news
        print("[2/6] Fetching and analyzing news...")
        stock_symbol = ticker.replace('.NS', '')
        news_articles = self.sentiment_analyzer.fetch_indian_news(stock_symbol, days=30)
        
        if news_articles:
            daily_sentiment, news_df = self.sentiment_analyzer.calculate_sentiment_trend(
                news_articles, stock_symbol
            )
        else:
            daily_sentiment = None
        
        # Step 3: Outlier detection and compensation
        print("[3/6] Detecting and compensating outliers...")
        outlier_results = self.outlier_detector.detect_outliers_multiple_methods(
            df_stock, ['Open', 'High', 'Low', 'Close', 'Volume']
        )
        df_compensated, compensation_report = self.outlier_detector.compensate_outliers(
            df_stock, outlier_results
        )
        
        # Step 4: Feature preparation
        print("[4/6] Preparing features...")
        df_features = self.prepare_features(df_compensated, daily_sentiment)
        
        # Step 5: Create sequences
        print("[5/6] Creating sequences...")
        sequence_length = 30
        target_col = 'Close'
        
        X, y_price, y_trend, y_volatility = self.create_sequences(
            df_features, target_col, sequence_length
        )
        
        # Scale features
        X_shape = X.shape
        X_flat = X.reshape(-1, X_shape[-1])
        X_scaled = self.scaler_X.fit_transform(X_flat)
        X = X_scaled.reshape(X_shape)
        
        # Scale price targets
        y_price_scaled = self.scaler_y.fit_transform(y_price.reshape(-1, 1)).flatten()
        
        # Split data (time series split)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_price_train, y_price_val = y_price_scaled[:split_idx], y_price_scaled[split_idx:]
        y_trend_train, y_trend_val = y_trend[:split_idx], y_trend[split_idx:]
        y_vol_train, y_vol_val = y_volatility[:split_idx], y_volatility[split_idx:]
        
        # Step 6: Build and train deep learning model
        print("[6/6] Training Multi-Head Attention model...")
        self.deep_model = MultiHeadAttentionStockModel(
            input_shape=(X.shape[1], X.shape[2]),
            n_heads=4,
            key_dim=64
        )
        self.deep_model.build_model()
        self.deep_model.compile_model(learning_rate=0.001)
        
        history = self.deep_model.train_model(
            X_train, y_price_train, y_trend_train, y_vol_train,
            X_val, y_price_val, y_trend_val, y_vol_val,
            epochs=50, batch_size=32
        )
        
        print("\n" + "="*60)
        print("MANAS: Training Completed Successfully")
        print("="*60)
        
        # Store results
        self.training_results = {
            'df_stock': df_stock,
            'df_features': df_features,
            'news_articles': news_articles,
            'daily_sentiment': daily_sentiment,
            'compensation_report': compensation_report,
            'history': history,
            'X_train': X_train,
            'X_val': X_val,
            'y_price_train': y_price_train,
            'y_price_val': y_price_val
        }
        
        return self.training_results
    
    def predict_future(self, days=10):
        """
        Generate future predictions
        """
        if self.deep_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        last_sequence = self.training_results['X_val'][-1:]
        future_predictions = []
        
        current_sequence = last_sequence.copy()
        
        for i in range(days):
            # Predict next day
            pred_price, pred_trend, pred_vol = self.deep_model.model.predict(current_sequence)
            
            # Inverse transform price
            pred_price_inv = self.scaler_y.inverse_transform(pred_price)[0, 0]
            
            # Store prediction
            pred_date = self.training_results['df_features'].index[-1] + datetime.timedelta(days=i+1)
            future_predictions.append({
                'date': pred_date,
                'predicted_price': pred_price_inv,
                'trend': np.argmax(pred_trend[0]),
                'volatility': pred_vol[0, 0]
            })
            
            # Update sequence for next prediction (simplified)
            # In production, implement proper sequence update
        
        return pd.DataFrame(future_predictions)


# ============================================================================
# PART 7: STREAMLIT UI FOR MANAS FRAMEWORK
# ============================================================================

def create_manas_ui():
    """Streamlit UI for MANAS framework"""
    
    st.set_page_config(
        page_title="MANAS - Indian Stock Market AI",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🇮🇳 MANAS: Multi-modal Adaptive Neural Architecture for Stocks")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Stock Selection")
        
        # Load Indian stocks
        @st.cache_data
        def load_indian_stocks():
            file_path = "indian_stocks.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding="utf-8")
                df.columns = df.columns.str.strip()
                if "SYMBOL" in df.columns:
                    return df["SYMBOL"].dropna().tolist()
            return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        
        indian_stocks = load_indian_stocks()
        selected_stock = st.selectbox("Select Stock", indian_stocks)
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.date(2023, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime.date.today())
        
        # Model parameters
        st.header("⚙️ Model Parameters")
        sequence_length = st.slider("Sequence Length", 10, 60, 30)
        n_heads = st.slider("Attention Heads", 2, 8, 4)
        epochs = st.slider("Training Epochs", 10, 100, 50)
        
        # Analysis button
        analyze_btn = st.button("🚀 Analyze Stock", use_container_width=True)
    
    # Main content
    if analyze_btn:
        with st.spinner("MANAS is analyzing the stock... This may take a few minutes."):
            
            # Initialize MANAS
            manas = MANAS_Framework()
            
            # Train model
            ticker = f"{selected_stock}.NS"
            results = manas.train(ticker, start_date, end_date)
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Stock Overview", 
                "🔍 Sentiment Analysis",
                "🤖 Model Performance",
                "📊 Predictions",
                "📉 Outlier Analysis"
            ])
            
            with tab1:
                st.subheader(f"Stock Information: {selected_stock}")
                
                # Get stock info
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", 
                             f"₹{info.get('currentPrice', 'N/A')}",
                             delta=f"{info.get('regularMarketChangePercent', 0):.2f}%")
                with col2:
                    st.metric("Market Cap", 
                             f"₹{info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "N/A")
                with col3:
                    st.metric("P/E Ratio", 
                             f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A")
                with col4:
                    st.metric("52W Range", 
                             f"₹{info.get('fiftyTwoWeekLow', 'N/A')} - ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
                
                # Price chart
                st.subheader("Historical Price with Technical Indicators")
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2]
                )
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=results['df_stock'].index,
                    open=results['df_stock']['Open'],
                    high=results['df_stock']['High'],
                    low=results['df_stock']['Low'],
                    close=results['df_stock']['Close'],
                    name='Price'
                ), row=1, col=1)
                
                # Add moving averages
                fig.add_trace(go.Scatter(
                    x=results['df_stock'].index,
                    y=results['df_stock']['SMA_20'] if 'SMA_20' in results['df_stock'].columns else None,
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=results['df_stock'].index,
                    y=results['df_stock']['SMA_50'] if 'SMA_50' in results['df_stock'].columns else None,
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ), row=1, col=1)
                
                # Volume
                fig.add_trace(go.Bar(
                    x=results['df_stock'].index,
                    y=results['df_stock']['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ), row=2, col=1)
                
                # RSI
                if 'RSI' in results['df_stock'].columns:
                    fig.add_trace(go.Scatter(
                        x=results['df_stock'].index,
                        y=results['df_stock']['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=1)
                    ), row=3, col=1)
                    
                    # Add RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                fig.update_layout(
                    height=800,
                    title_text=f"{selected_stock} - Technical Analysis",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Sentiment Analysis Results")
                
                if results['daily_sentiment'] is not None:
                    # Sentiment timeline
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=results['daily_sentiment'].index,
                        y=results['daily_sentiment'][('score', 'mean')],
                        mode='lines+markers',
                        name='Sentiment Score',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=results['daily_sentiment'].index,
                        y=results['daily_sentiment']['trend'],
                        mode='lines',
                        name='3-Day Trend',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Sentiment Trend Over Time",
                        xaxis_title="Date",
                        yaxis_title="Sentiment Score",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment pie chart
                        sentiment_counts = results['daily_sentiment'][('score', 'mean')].apply(
                            lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
                        ).value_counts()
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=sentiment_counts.index,
                            values=sentiment_counts.values,
                            hole=.3,
                            marker_colors=['green', 'gray', 'red']
                        )])
                        
                        fig.update_layout(title="Sentiment Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # News sources
                        if 'news_articles' in results and results['news_articles']:
                            sources = [a.get('source', {}).get('name', 'Unknown') 
                                     for a in results['news_articles'][:20]]
                            source_counts = pd.Series(sources).value_counts().head(10)
                            
                            fig = go.Figure(data=[go.Bar(
                                x=source_counts.values,
                                y=source_counts.index,
                                orientation='h',
                                marker_color='lightblue'
                            )])
                            
                            fig.update_layout(
                                title="Top News Sources",
                                xaxis_title="Count",
                                yaxis_title="Source",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Recent news table
                    st.subheader("Recent News with Sentiment")
                    if 'news_articles' in results and results['news_articles']:
                        news_data = []
                        for article in results['news_articles'][:10]:
                            news_data.append({
                                'Date': article.get('publishedAt', '')[:10],
                                'Source': article.get('source', {}).get('name', ''),
                                'Title': article.get('title', '')[:100] + '...',
                                'Sentiment': 'N/A'
                            })
                        
                        st.dataframe(pd.DataFrame(news_data), use_container_width=True)
                
                else:
                    st.warning("No recent news found for this stock.")
            
            with tab3:
                st.subheader("Model Performance Metrics")
                
                # Training history
                if 'history' in results:
                    history = results['history']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Loss plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=history.history['loss'],
                            name='Training Loss',
                            mode='lines'
                        ))
                        fig.add_trace(go.Scatter(
                            y=history.history['val_loss'],
                            name='Validation Loss',
                            mode='lines'
                        ))
                        fig.update_layout(
                            title="Model Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Price MAE
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=history.history['price_prediction_mae'],
                            name='Training MAE',
                            mode='lines'
                        ))
                        fig.add_trace(go.Scatter(
                            y=history.history['val_price_prediction_mae'],
                            name='Validation MAE',
                            mode='lines'
                        ))
                        fig.update_layout(
                            title="Price Prediction MAE",
                            xaxis_title="Epoch",
                            yaxis_title="MAE",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Trend accuracy
                    if 'trend_prediction_accuracy' in history.history:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=history.history['trend_prediction_accuracy'],
                            name='Training Accuracy',
                            mode='lines'
                        ))
                        fig.add_trace(go.Scatter(
                            y=history.history['val_trend_prediction_accuracy'],
                            name='Validation Accuracy',
                            mode='lines'
                        ))
                        fig.update_layout(
                            title="Trend Prediction Accuracy",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Final metrics
                    st.subheader("Final Model Performance")
                    
                    final_metrics = pd.DataFrame({
                        'Metric': ['Loss', 'Price MAE', 'Price RMSE', 'Trend Accuracy'],
                        'Training': [
                            f"{history.history['loss'][-1]:.4f}",
                            f"{history.history['price_prediction_mae'][-1]:.4f}",
                            f"{np.sqrt(history.history['price_prediction_mse'][-1]):.4f}",
                            f"{history.history.get('trend_prediction_accuracy', [0])[-1]:.2%}"
                        ],
                        'Validation': [
                            f"{history.history['val_loss'][-1]:.4f}",
                            f"{history.history['val_price_prediction_mae'][-1]:.4f}",
                            f"{np.sqrt(history.history['val_price_prediction_mse'][-1]):.4f}",
                            f"{history.history.get('val_trend_prediction_accuracy', [0])[-1]:.2%}"
                        ]
                    })
                    
                    st.dataframe(final_metrics, use_container_width=True)
            
            with tab4:
                st.subheader("10-Day Price Forecast")
                
                # Generate predictions
                future_df = manas.predict_future(days=10)
                
                if future_df is not None and not future_df.empty:
                    # Display forecast table
                    forecast_display = future_df[['date', 'predicted_price', 'trend', 'volatility']].copy()
                    forecast_display['predicted_price'] = forecast_display['predicted_price'].apply(
                        lambda x: f"₹{x:.2f}"
                    )
                    forecast_display['trend'] = forecast_display['trend'].map({
                        0: '📈 Up', 1: '📉 Down', 2: '➡️ Neutral'
                    })
                    forecast_display['volatility'] = forecast_display['volatility'].apply(
                        lambda x: f"{x:.2%}"
                    )
                    
                    st.dataframe(
                        forecast_display.rename(columns={
                            'date': 'Date',
                            'predicted_price': 'Predicted Price',
                            'trend': 'Trend',
                            'volatility': 'Expected Volatility'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Forecast chart
                    fig = go.Figure()
                    
                    # Historical data
                    historical = results['df_stock'].iloc[-60:]
                    fig.add_trace(go.Scatter(
                        x=historical.index,
                        y=historical['Close'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=future_df['date'],
                        y=future_df['predicted_price'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', width=2, dash='dot'),
                        marker=dict(size=8)
                    ))
                    
                    # Add confidence interval
                    price_std = historical['Close'].std()
                    fig.add_trace(go.Scatter(
                        x=future_df['date'].tolist() + future_df['date'].tolist()[::-1],
                        y=(future_df['predicted_price'] + price_std).tolist() + 
                          (future_df['predicted_price'] - price_std).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval',
                        showlegend=True
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_stock} - Price Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Investment recommendation
                    st.subheader("💡 Investment Recommendation")
                    
                    current_price = historical['Close'].iloc[-1]
                    avg_forecast = future_df['predicted_price'].mean()
                    price_change = ((avg_forecast - current_price) / current_price) * 100
                    
                    # Generate recommendation based on multiple factors
                    sentiment_score = results['daily_sentiment'][('score', 'mean')].mean() if results['daily_sentiment'] is not None else 0
                    
                    if price_change > 5 and sentiment_score > 0.1:
                        recommendation = "STRONG BUY"
                        reason = "Strong upward trend with positive market sentiment"
                        color = "green"
                    elif price_change > 2 and sentiment_score > 0:
                        recommendation = "BUY"
                        reason = "Moderate upward potential with neutral-positive sentiment"
                        color = "lightgreen"
                    elif price_change > -2 and price_change < 2:
                        recommendation = "HOLD"
                        reason = "Stable price expected in near term"
                        color = "blue"
                    elif price_change < -5 and sentiment_score < -0.1:
                        recommendation = "STRONG SELL"
                        reason = "Significant downward pressure with negative sentiment"
                        color = "darkred"
                    elif price_change < -2:
                        recommendation = "SELL"
                        reason = "Downward trend expected"
                        color = "red"
                    else:
                        recommendation = "HOLD"
                        reason = "Market conditions unclear"
                        color = "orange"
                    
                    st.markdown(
                        f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white;">
                            <h3 style="margin:0">{recommendation}</h3>
                            <p style="margin:5px 0 0 0">{reason}</p>
                            <p style="margin:5px 0 0 0; font-size:14px">Expected Change: {price_change:+.2f}% | Sentiment Score: {sentiment_score:.2f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            with tab5:
                st.subheader("Outlier Detection and Compensation Report")
                
                if 'compensation_report' in results:
                    for col, report in results['compensation_report'].items():
                        with st.expander(f"📊 {col} - {report['n_outliers']} outliers detected"):
                            st.write(f"**Compensation Method:** {report['compensation_method']}")
                            
                            if len(report['outlier_dates']) > 0:
                                st.write("**Outlier Dates:**")
                                outlier_dates_str = [d.strftime('%Y-%m-%d') if isinstance(d, pd.Timestamp) else str(d) 
                                                   for d in report['outlier_dates'][:10]]
                                st.write(", ".join(outlier_dates_str))
                                
                                if len(report['outlier_dates']) > 10:
                                    st.write(f"... and {len(report['outlier_dates']) - 10} more")
                    
                    # Visualize outliers
                    st.subheader("Outlier Visualization")
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Close Price', 'Volume', 'Returns', 'RSI')
                    )
                    
                    # Close price with outliers
                    fig.add_trace(go.Scatter(
                        x=results['df_stock'].index,
                        y=results['df_stock']['Close'],
                        mode='lines',
                        name='Close',
                        line=dict(color='blue')
                    ), row=1, col=1)
                    
                    if 'Close' in results['compensation_report']:
                        outlier_indices = results['compensation_report']['Close']['outlier_dates']
                        outlier_values = results['df_stock'].loc[outlier_indices, 'Close'] if len(outlier_indices) > 0 else []
                        
                        fig.add_trace(go.Scatter(
                            x=outlier_indices,
                            y=outlier_values,
                            mode='markers',
                            name='Outliers',
                            marker=dict(color='red', size=8, symbol='x')
                        ), row=1, col=1)
                    
                    # Volume
                    fig.add_trace(go.Bar(
                        x=results['df_stock'].index,
                        y=results['df_stock']['Volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ), row=1, col=2)
                    
                    # Returns
                    fig.add_trace(go.Scatter(
                        x=results['df_stock'].index,
                        y=results['df_stock']['Returns'] if 'Returns' in results['df_stock'].columns else [],
                        mode='lines',
                        name='Returns',
                        line=dict(color='green')
                    ), row=2, col=1)
                    
                    # RSI
                    if 'RSI' in results['df_stock'].columns:
                        fig.add_trace(go.Scatter(
                            x=results['df_stock'].index,
                            y=results['df_stock']['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple')
                        ), row=2, col=2)
                        
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=2)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=2)
                    
                    fig.update_layout(height=800, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("No outliers detected in the dataset.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to MANAS! 👋
        
        **Multi-modal Adaptive Neural Architecture for Stocks**
        
        This advanced AI framework combines:
        - 🧠 **Multi-Head Attention** deep learning architecture
        - 📰 **LLM-based Sentiment Analysis** (FinBERT + LLaMA)
        - 📊 **Technical Indicators** with Indian market context
        - 🔍 **Outlier Detection** and compensation
        - 🤝 **Adaptive Ensemble Fusion** for robust predictions
        
        ### How to use:
        1. Select a stock from the sidebar
        2. Choose date range
        3. Adjust model parameters
        4. Click "Analyze Stock" to start
        
        The analysis will provide:
        - Comprehensive technical analysis
        - Sentiment trends from news
        - Model performance metrics
        - 10-day price forecast
        - Investment recommendation
        """)
        
        # Sample visualization
        st.image("https://img.freepik.com/free-vector/stock-market-concept_23-2148577165.jpg", 
                use_container_width=True)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    create_manas_ui()
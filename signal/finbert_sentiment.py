"""
FinBERT Sentiment Analysis Module
Provides sentiment signals from news headlines and earnings transcripts.
Uses FinBERT model fine-tuned on financial text.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import re
import json
import os
from dataclasses import dataclass


@dataclass
class SentimentSignal:
    """Sentiment signal for a ticker."""
    ticker: str
    sentiment_score: float  # -1.0 to +1.0
    confidence: float       # 0.0 to 1.0
    source: str             # 'news', 'transcript', 'social'
    timestamp: datetime
    num_mentions: int
    exponential_decay: float  # Time-decayed score


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    """
    
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda:0" if self._check_cuda() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Lazy-load FinBERT model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            
            print(f"[FinBERT] Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            if self.device.startswith('cuda'):
                self.model = self.model.to(self.device)
            
            # Create sentiment pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.startswith('cuda') else -1
            )
            print(f"[FinBERT] Model loaded successfully")
            
        except ImportError:
            print("[FinBERT] Warning: transformers not installed. Using mock sentiment.")
            self.pipeline = None
        except Exception as e:
            print(f"[FinBERT] Error loading model: {e}")
            self.pipeline = None
    
    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            (sentiment_score, confidence) where sentiment is -1 to +1
        """
        if self.pipeline is None or not text:
            return 0.0, 0.0
        
        try:
            # Truncate long texts
            text = text[:512] if len(text) > 512 else text
            
            result = self.pipeline(text)[0]
            label = result['label']
            confidence = result['score']
            
            # Map to -1 to +1 scale
            if 'positive' in label.lower():
                sentiment = confidence
            elif 'negative' in label.lower():
                sentiment = -confidence
            else:
                sentiment = 0.0
            
            return sentiment, confidence
            
        except Exception as e:
            print(f"[FinBERT] Analysis error: {e}")
            return 0.0, 0.0
    
    def analyze_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Analyze sentiment for multiple texts."""
        if self.pipeline is None:
            return [(0.0, 0.0)] * len(texts)
        
        # Truncate texts
        texts = [t[:512] if len(t) > 512 else t for t in texts]
        
        try:
            results = self.pipeline(texts, batch_size=16)
            sentiments = []
            
            for result in results:
                label = result['label']
                confidence = result['score']
                
                if 'positive' in label.lower():
                    sentiment = confidence
                elif 'negative' in label.lower():
                    sentiment = -confidence
                else:
                    sentiment = 0.0
                
                sentiments.append((sentiment, confidence))
            
            return sentiments
            
        except Exception as e:
            print(f"[FinBERT] Batch analysis error: {e}")
            return [(0.0, 0.0)] * len(texts)


class NewsSentimentFetcher:
    """
    Fetches and analyzes news sentiment for tickers.
    """
    
    def __init__(self, api_key: Optional[str] = None, analyzer: Optional[FinBERTSentimentAnalyzer] = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.analyzer = analyzer or FinBERTSentimentAnalyzer()
        self.cache: Dict[str, pd.DataFrame] = {}
        
    def fetch_news_headlines(self, ticker: str, days: int = 7) -> List[Dict]:
        """
        Fetch news headlines for a ticker.
        
        Note: This is a placeholder. In production, integrate with:
        - NewsAPI (newsapi.org)
        - Bloomberg API
        - Refinitiv
        - Alpaca News API
        """
        if not self.api_key:
            # Return mock data for demonstration
            return self._mock_news(ticker, days)
        
        # Example NewsAPI integration
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': ticker,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if data.get('status') == 'ok':
                return [
                    {
                        'title': article['title'],
                        'description': article.get('description', ''),
                        'published_at': article['publishedAt'],
                        'source': article['source']['name']
                    }
                    for article in data.get('articles', [])
                ]
            else:
                print(f"[News] API error: {data.get('message')}")
                return []
                
        except Exception as e:
            print(f"[News] Fetch error: {e}")
            return []
    
    def _mock_news(self, ticker: str, days: int) -> List[Dict]:
        """Generate mock news for testing."""
        import random
        
        templates = [
            (f"{ticker} beats earnings expectations", 0.8),
            (f"{ticker} misses revenue target", -0.6),
            (f"Analyst upgrades {ticker} to buy", 0.7),
            (f"{ticker} announces stock buyback program", 0.5),
            (f"{ticker} faces regulatory scrutiny", -0.7),
            (f"{ticker} expands into new markets", 0.4),
            (f"{ticker} CEO steps down", -0.5),
            (f"{ticker} partners with tech giant", 0.6),
        ]
        
        news = []
        for i in range(min(5, days)):
            template, bias = random.choice(templates)
            date = datetime.now() - timedelta(days=i)
            news.append({
                'title': template,
                'description': f"Details about {ticker}...",
                'published_at': date.isoformat(),
                'source': 'Mock Financial News'
            })
        
        return news
    
    def get_ticker_sentiment(self, ticker: str, lookback_days: int = 7) -> SentimentSignal:
        """
        Get aggregated sentiment for a ticker.
        """
        # Fetch news
        news_items = self.fetch_news_headlines(ticker, lookback_days)
        
        if not news_items:
            return SentimentSignal(
                ticker=ticker,
                sentiment_score=0.0,
                confidence=0.0,
                source='news',
                timestamp=datetime.now(),
                num_mentions=0,
                exponential_decay=0.0
            )
        
        # Analyze sentiment for each headline
        texts = [item['title'] + ' ' + item.get('description', '') for item in news_items]
        sentiments = self.analyzer.analyze_batch(texts)
        
        # Calculate time-decayed weighted sentiment
        current_time = datetime.now()
        weighted_scores = []
        total_weight = 0.0
        
        for i, (item, (sentiment, confidence)) in enumerate(zip(news_items, sentiments)):
            # Parse timestamp
            try:
                item_time = datetime.fromisoformat(item['published_at'].replace('Z', '+00:00'))
                hours_ago = (current_time - item_time).total_seconds() / 3600
            except:
                hours_ago = i * 24  # Fallback
            
            # Exponential decay: weight = exp(-hours/48) => half-life of 2 days
            decay = np.exp(-hours_ago / 48)
            weight = decay * confidence  # Higher confidence = more weight
            
            weighted_scores.append(sentiment * weight)
            total_weight += weight
        
        # Aggregate
        if total_weight > 0:
            avg_sentiment = sum(weighted_scores) / total_weight
            avg_confidence = sum(s for _, s in sentiments) / len(sentiments)
            exponential_decay = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
        else:
            avg_sentiment = 0.0
            avg_confidence = 0.0
            exponential_decay = 0.0
        
        return SentimentSignal(
            ticker=ticker,
            sentiment_score=avg_sentiment,
            confidence=avg_confidence,
            source='news',
            timestamp=datetime.now(),
            num_mentions=len(news_items),
            exponential_decay=exponential_decay
        )
    
    def get_universe_sentiment(self, tickers: List[str], lookback_days: int = 7) -> pd.DataFrame:
        """
        Get sentiment for multiple tickers.
        
        Returns DataFrame with sentiment metrics.
        """
        results = []
        
        for ticker in tickers:
            signal = self.get_ticker_sentiment(ticker, lookback_days)
            results.append({
                'Ticker': signal.ticker,
                'Sentiment_Score': signal.sentiment_score,
                'Confidence': signal.confidence,
                'Num_Mentions': signal.num_mentions,
                'Exp_Decay_Score': signal.exponential_decay,
                'Timestamp': signal.timestamp
            })
        
        df = pd.DataFrame(results).set_index('Ticker')
        return df


class EarningsSentimentExtractor:
    """
    Extracts sentiment from earnings call transcripts.
    """
    
    def __init__(self, analyzer: Optional[FinBERTSentimentAnalyzer] = None):
        self.analyzer = analyzer or FinBERTSentimentAnalyzer()
    
    def analyze_transcript(self, transcript_text: str, ticker: str) -> SentimentSignal:
        """
        Analyze earnings call transcript.
        
        Breaks transcript into segments and analyzes each.
        """
        if not transcript_text:
            return SentimentSignal(
                ticker=ticker,
                sentiment_score=0.0,
                confidence=0.0,
                source='transcript',
                timestamp=datetime.now(),
                num_mentions=1,
                exponential_decay=0.0
            )
        
        # Split into segments (paragraphs)
        segments = [s.strip() for s in transcript_text.split('\n\n') if len(s.strip()) > 50]
        
        if not segments:
            segments = [transcript_text[:1000]]  # Fallback to first 1000 chars
        
        # Analyze each segment
        sentiments = self.analyzer.analyze_batch(segments)
        
        # Weight by segment position (management discussion gets higher weight)
        weights = [1.0 / (i + 1) for i in range(len(segments))]
        total_weight = sum(weights)
        
        weighted_sentiment = sum(s * w for (s, _), w in zip(sentiments, weights)) / total_weight
        avg_confidence = sum(c for _, c in sentiments) / len(sentiments)
        
        return SentimentSignal(
            ticker=ticker,
            sentiment_score=weighted_sentiment,
            confidence=avg_confidence,
            source='transcript',
            timestamp=datetime.now(),
            num_mentions=len(segments),
            exponential_decay=weighted_sentiment
        )


# Convenience functions
def get_sentiment_signals(tickers: List[str], lookback_days: int = 7) -> pd.DataFrame:
    """
    Quick function to get sentiment signals for a list of tickers.
    """
    fetcher = NewsSentimentFetcher()
    return fetcher.get_universe_sentiment(tickers, lookback_days)


def combine_with_kronos(kronos_signals: pd.DataFrame, sentiment_signals: pd.DataFrame,
                        kronos_weight: float = 0.6, sentiment_weight: float = 0.4) -> pd.DataFrame:
    """
    Combine Kronos price signals with sentiment signals.
    
    Args:
        kronos_signals: DataFrame with 'Predicted_Return' column
        sentiment_signals: DataFrame with 'Sentiment_Score' column
        kronos_weight: Weight for Kronos signals (0-1)
        sentiment_weight: Weight for sentiment signals (0-1)
    
    Returns:
        DataFrame with combined signals
    """
    # Align indices
    common_tickers = kronos_signals.index.intersection(sentiment_signals.index)
    
    combined = pd.DataFrame(index=common_tickers)
    
    # Normalize Kronos signals to -1 to +1 scale
    kronos_norm = kronos_signals.loc[common_tickers, 'Predicted_Return']
    kronos_norm = (kronos_norm - kronos_norm.mean()) / (kronos_norm.std() + 1e-5)
    kronos_norm = np.clip(kronos_norm, -3, 3) / 3  # Scale to roughly -1 to +1
    
    # Get sentiment scores
    sentiment = sentiment_signals.loc[common_tickers, 'Sentiment_Score']
    
    # Weighted combination
    combined['Kronos_Signal'] = kronos_norm
    combined['Sentiment_Signal'] = sentiment
    combined['Combined_Signal'] = kronos_weight * kronos_norm + sentiment_weight * sentiment
    combined['Confidence'] = sentiment_signals.loc[common_tickers, 'Confidence']
    
    return combined.sort_values('Combined_Signal', ascending=False)

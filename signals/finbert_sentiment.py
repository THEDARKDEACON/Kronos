"""
FinBERT Sentiment Analysis Module
Provides sentiment signals from news headlines and earnings transcripts.
Uses FinBERT model fine-tuned on financial text.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import time
from urllib.parse import quote
import requests
import re
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta


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
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None):
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
        """Lazy-load FinBERT model with numpy 2.x compatibility. Try ONNX first."""
        # Try ONNX first (numpy 2.x compatible, no pickle issues)
        if self._try_load_onnx():
            return
        
        # Fall back to PyTorch/Transformers
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            import os
            
            print(f"[FinBERT] Loading {self.model_name}...")
            
            # Force safetensors format to avoid numpy pickle issues
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_safetensors=True
            )
            
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    torch_dtype='auto'
                )
            except Exception as e:
                # Fallback: try without safetensors if not available
                print(f"[FinBERT] Safetensors failed, trying pickle format...")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    use_safetensors=False
                )
            
            if self.device.startswith('cuda') and self.model is not None:
                self.model = self.model.to(self.device)
            
            # Create sentiment pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.startswith('cuda') else -1,
                batch_size=16
            )
            print(f"[FinBERT] Model loaded successfully")
            self._use_onnx = False
            
        except ImportError:
            print("[FinBERT] Warning: transformers not installed. Using mock sentiment.")
            self.pipeline = None
            self._use_onnx = False
        except Exception as e:
            print(f"[FinBERT] Error loading model: {e}")
            print("[FinBERT] Falling back to neutral sentiment (0.0)")
            self.pipeline = None
            self._use_onnx = False
    
    def _try_load_onnx(self) -> bool:
        """Try to load ONNX model. Returns True if successful."""
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            from pathlib import Path
            
            # Check for ONNX model in models/onnx directory
            model_short = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
            onnx_path = Path(f"./models/onnx/{model_short}_quantized.onnx")
            tokenizer_path = Path(f"./models/onnx/{model_short}_tokenizer")
            
            if not onnx_path.exists():
                # Try non-quantized version
                onnx_path = Path(f"./models/onnx/{model_short}.onnx")
            
            if not onnx_path.exists() or not tokenizer_path.exists():
                return False
            
            print(f"[FinBERT] Loading ONNX model: {onnx_path}")
            
            # Create inference session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.startswith('cuda') else ['CPUExecutionProvider']
            self._onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            
            print(f"[FinBERT] ONNX model loaded successfully (providers: {self._onnx_session.get_providers()})")
            self._use_onnx = True
            return True
            
        except ImportError:
            # onnxruntime not installed
            return False
        except Exception as e:
            print(f"[FinBERT] ONNX load failed: {e}")
            return False
    
    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            (sentiment_score, confidence) where sentiment is -1 to +1
        """
        if not text:
            return 0.0, 0.0
        
        # Use ONNX if available
        if hasattr(self, '_use_onnx') and self._use_onnx and hasattr(self, '_onnx_session'):
            return self._analyze_text_onnx(text)
        
        if self.pipeline is None:
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
    
    def _analyze_text_onnx(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment using ONNX runtime."""
        try:
            import numpy as np
            
            # Truncate long texts
            text = text[:512] if len(text) > 512 else text
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="np", max_length=512, padding="max_length", truncation=True)
            
            # Run inference
            ort_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
            
            ort_outputs = self._onnx_session.run(None, ort_inputs)
            logits = ort_outputs[0]
            
            # Get prediction
            # M-10 Fix: Numerically stable softmax to avoid np.exp overflow
            shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
            probs = np.exp(shifted_logits) / np.sum(np.exp(shifted_logits), axis=-1, keepdims=True)
            pred_idx = np.argmax(probs, axis=-1)[0]
            confidence = float(np.max(probs))
            
            # Map to sentiment
            labels = ["negative", "neutral", "positive"]
            label = labels[pred_idx]
            
            if label == "positive":
                sentiment = confidence
            elif label == "negative":
                sentiment = -confidence
            else:
                sentiment = 0.0
            
            return sentiment, confidence
            
        except Exception as e:
            print(f"[FinBERT] ONNX analysis error: {e}")
            return 0.0, 0.0
    
    def analyze_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Analyze sentiment for multiple texts."""
        # Use ONNX if available
        if hasattr(self, '_use_onnx') and self._use_onnx and hasattr(self, '_onnx_session'):
            return [self._analyze_text_onnx(t) for t in texts]
        
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
    Fetches and analyzes news sentiment for tickers using NewsAPI.
    Includes rate limiting and caching for production use.
    """
    
    def __init__(self, api_key: Optional[str] = None, analyzer: Optional[FinBERTSentimentAnalyzer] = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY') or os.getenv('NEWSAPI_KEY')
        self.analyzer = analyzer or FinBERTSentimentAnalyzer()
        self.cache: Dict[str, pd.DataFrame] = {}
        self._last_request_time = 0
        self._min_request_interval = 1.2  # NewsAPI free tier: 100 requests/day = ~1 per 14min
        # For paid tier: reduce to 0.1 (10 req/sec)
        
    def fetch_news_headlines(self, ticker: str, days: int = 7) -> List[Dict]:
        """
        Fetch news headlines for a ticker using NewsAPI.

        Returns an empty list when no API key is configured or on API errors
        (sentiment falls back to neutral 0.0).
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of news articles with headline, source, date
        """
        if not self.api_key:
            return []

        try:
            return self._fetch_from_newsapi(ticker, days)
        except Exception as e:
            print(f"[NewsAPI] Error fetching for {ticker}: {e}")
            return []
    
    def _fetch_from_newsapi(self, ticker: str, days: int) -> List[Dict]:
        """
        Fetch news from NewsAPI with rate limiting.
        
        NewsAPI Free Tier: 100 requests/day
        NewsAPI Paid Tier: 10 requests/second
        """
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            print(f"[NewsAPI] Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        # Build query - search for ticker symbol and company name
        query = f"{ticker} stock OR {ticker} earnings OR {ticker} finance"
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            'to': datetime.now().strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 20,  # Max 100 for paid, 20 for free
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            self._last_request_time = time.time()
            
            if response.status_code == 429:
                print("[NewsAPI] Rate limit exceeded.")
                return []
            
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                print(f"[NewsAPI] Fetched {len(articles)} articles for {ticker}")
                
                return [
                    {
                        'title': article['title'],
                        'description': article.get('description', ''),
                        'published_at': article['publishedAt'],
                        'source': article['source']['name'],
                        'url': article.get('url', '')
                    }
                    for article in articles
                    if article.get('title')  # Filter out empty titles
                ]
            else:
                error_msg = data.get('message', 'Unknown error')
                print(f"[NewsAPI] API error: {error_msg}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"[NewsAPI] Request failed: {e}")
            return []
        except Exception as e:
            print(f"[NewsAPI] Unexpected error: {e}")
            return []
    
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
        
        # L-12 Fix: A strict 1/(i+1) positional decay assumes the first paragraph
        # is the most important, but in earnings calls the first 2-3 paragraphs are
        # typically legal safe-harbor disclaimers. We use a gentler decay that gives
        # equal weight to the first 5 segments (prepared remarks), then decays.
        weights = [1.0 if i < 5 else 5.0 / (i + 1) for i in range(len(segments))]
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

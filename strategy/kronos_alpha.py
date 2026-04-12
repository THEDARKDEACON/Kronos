import sys
import os
import pandas as pd
from typing import Dict, List

# The Kronos Engine modules are now located natively in the same repo, so no path append is necessary

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    import torch
    KRONOS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Kronos modules from local repo. Ensure path is correct. {e}")
    KRONOS_AVAILABLE = False
    torch = None

class KronosAlphaGenerator:
    def __init__(self, model_size="small", max_context=512, device=None):
        """
        Initializes the Kronos predictor. Automatically mounts to GPU if found.
        """
        import torch
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Kronos Engine dynamically mounting to Hardware: {self.device}")
            
        self.model_size = model_size
        self.max_context = max_context
        self.predictor = None
        self.batch_size = 16  # Conservative default for memory-constrained GPUs
        
        if KRONOS_AVAILABLE:
            print(f"Loading Kronos-{model_size}...")
            # We fetch from HuggingFace, but you could load local checkpoints if available
            self.tokenizer = KronosTokenizer.from_pretrained(f"NeoQuasar/Kronos-Tokenizer-base")
            self.model = Kronos.from_pretrained(f"NeoQuasar/Kronos-{model_size}")
            
            # Send to device BEFORE creating predictor
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode for inference
            
            # Check GPU memory and adjust settings
            if self.device.startswith('cuda'):
                gpu_memory_gb = self._get_gpu_memory_gb()
                print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
                
                # Adjust batch size based on available memory
                if gpu_memory_gb < 4:
                    self.batch_size = 8
                    print(f"Low VRAM detected. Using conservative batch_size={self.batch_size}")
                    # Skip torch.compile on low memory GPUs (adds overhead)
                elif gpu_memory_gb < 8:
                    self.batch_size = 16
                    print(f"Using batch_size={self.batch_size}")
                    # Enable torch.compile only on GPUs with sufficient memory
                    if hasattr(torch, 'compile'):
                        print("Enabling torch.compile() for optimized inference...")
                        self.model = torch.compile(self.model, mode='reduce-overhead')
                else:
                    self.batch_size = 32
                    print(f"High VRAM available. Using batch_size={self.batch_size}")
                    if hasattr(torch, 'compile'):
                        print("Enabling torch.compile() for optimized inference...")
                        self.model = torch.compile(self.model, mode='reduce-overhead')
            
            # Enable mixed precision for memory efficiency
            self.use_amp = self.device.startswith('cuda')
            if self.use_amp:
                print("Enabling Automatic Mixed Precision (FP16) for memory efficiency...")
            
            # Create predictor after model is on device
            self.predictor = KronosPredictor(self.model, self.tokenizer, max_context=self.max_context, device=self.device)
            
    def _get_gpu_memory_gb(self) -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return total_memory / (1024**3)

    def generate_signals(self, ohlcv_data: Dict[str, pd.DataFrame], lookback: int = 400, pred_len: int = 10) -> pd.DataFrame:
        """
        Takes the dictionary of OHLCV DataFrames and runs predict_batch.
        Returns a DataFrame mapping Ticker -> Predicted Return
        """
        if not KRONOS_AVAILABLE or self.predictor is None:
            print("Kronos not loaded. Returning mock signals.")
            return self._mock_signals(ohlcv_data.keys())

        # Prepare lists for batching
        tickers = list(ohlcv_data.keys())
        df_list = []
        x_timestamp_list = []
        y_timestamp_list = []

        # Kronos predict_batch requires all matrices to be the EXACT same length
        max_possible_len = max(len(df) for df in ohlcv_data.values())
        strict_len = int(max_possible_len * 0.95) # Tolerate missing holidays, drop IPOs
        
        for ticker in tickers:
            df = ohlcv_data[ticker]
            if len(df) < strict_len:
                print(f"Skipping {ticker} (len {len(df)} < {strict_len} required for tensor batching).")
                continue
                
            x_df = df.iloc[-strict_len:].reset_index(drop=True)
            
            # Predict arbitrary future timestamps (e.g. next N days)
            last_timestamp = x_df['timestamps'].iloc[-1]
            future_timestamps = [last_timestamp + pd.Timedelta(days=i) for i in range(1, pred_len + 1)]
            
            df_list.append(x_df[['open', 'high', 'low', 'close', 'volume', 'amount']])
            x_timestamp_list.append(x_df['timestamps'])
            y_timestamp_list.append(pd.Series(future_timestamps))

        if not df_list:
            return pd.DataFrame()

        print(f"Running Kronos batch prediction on {len(df_list)} tickers (batch_size={self.batch_size})...")
        
        # Chunked inference to prevent OOM on memory-constrained GPUs
        pred_df_list = []
        num_tickers = len(df_list)
        
        for i in range(0, num_tickers, self.batch_size):
            chunk_end = min(i + self.batch_size, num_tickers)
            print(f"   Processing batch {i//self.batch_size + 1}/{(num_tickers-1)//self.batch_size + 1} (tickers {i+1}-{chunk_end})...")
            
            # Extract chunk
            chunk_df_list = df_list[i:chunk_end]
            chunk_x_timestamps = x_timestamp_list[i:chunk_end]
            chunk_y_timestamps = y_timestamp_list[i:chunk_end]
            
            # Run inference on chunk
            with torch.cuda.amp.autocast(enabled=self.use_amp):  # Mixed precision
                chunk_preds = self.predictor.predict_batch(
                    df_list=chunk_df_list,
                    x_timestamp_list=chunk_x_timestamps,
                    y_timestamp_list=chunk_y_timestamps,
                    pred_len=pred_len,
                    verbose=False
                )
            
            pred_df_list.extend(chunk_preds)
            
            # Clear GPU cache between batches to prevent fragmentation
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
        
        print(f"Completed inference for {len(pred_df_list)} tickers")

        # Calculate expected momentum (Percentage return from current to last predicted close)
        signals = []
        for i, pred_df in enumerate(pred_df_list):
            ticker = tickers[i]
            x_df = df_list[i]
            
            current_close = x_df['close'].iloc[-1]
            predicted_close = pred_df['close'].iloc[-1]
            
            expected_return = (predicted_close - current_close) / current_close
            
            signals.append({
                'Ticker': ticker,
                'Predicted_Return': expected_return
            })
            
        signal_df = pd.DataFrame(signals).set_index('Ticker')
        return signal_df.sort_values(by='Predicted_Return', ascending=False)
        
    def _mock_signals(self, tickers):
        # Mocks a random momentum signal if Kronos isn't actually installed
        import numpy as np
        signals = []
        for ticker in tickers:
            signals.append({'Ticker': ticker, 'Predicted_Return': np.random.normal(0, 0.05)})
        return pd.DataFrame(signals).set_index('Ticker').sort_values(by='Predicted_Return', ascending=False)

if __name__ == "__main__":
    import numpy as np
    # Test block
    gen = KronosAlphaGenerator()
    
    # Create mock OHLCV dict
    dfs = {}
    for t in ['AAPL', 'MSFT', 'XOM']:
        dates = pd.date_range("2025-01-01", periods=100, freq='D')
        dfs[t] = pd.DataFrame({
            'open': np.linspace(100, 110, 100),
            'high': np.linspace(105, 115, 100),
            'low': np.linspace(95, 105, 100),
            'close': np.linspace(102, 112, 100),
            'volume': 1000 * np.ones(100),
            'amount': np.zeros(100),
            'timestamps': dates
        })
        
    signals = gen.generate_signals(dfs, lookback=100, pred_len=5)
    print("\nPredicted Signals:")
    print(signals)

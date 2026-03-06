"""
Quick test to debug Chronos wrapper
"""
import torch
import numpy as np
import pandas as pd
from core.model_loader import load_chronos_predictor

# Load predictor
print("Loading Chronos predictor...")
predictor = load_chronos_predictor()

# Create simple test data
print("\nCreating test data...")
dates = pd.date_range('2020-01-01', periods=100, freq='D')
prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

df = pd.DataFrame({
    'datetime': dates,
    'open': prices * 0.99,
    'high': prices * 1.01,
    'low': prices * 0.98,
    'close': prices
})

print(f"Test data shape: {df.shape}")
print(f"Close prices: {df['close'].head()}")

# Test direct pipeline call first
print("\nTesting direct pipeline call...")
try:
    import torch
    test_context = df.iloc[:80]['close'].values
    context_tensor = torch.tensor(test_context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(f"Input shape: {context_tensor.shape}")
    
    forecast = predictor.pipeline.predict(
        inputs=context_tensor,
        prediction_length=12
    )
    
    print(f"Forecast type: {type(forecast)}")
    if isinstance(forecast, list):
        print(f"Forecast is list with {len(forecast)} elements")
        print(f"First element type: {type(forecast[0])}")
        if len(forecast) > 0:
            print(f"First element shape: {forecast[0].shape if hasattr(forecast[0], 'shape') else 'N/A'}")
    else:
        print(f"Forecast shape: {forecast.shape if hasattr(forecast, 'shape') else 'N/A'}")
    
except Exception as e:
    print(f"Direct pipeline test failed: {e}")
    import traceback
    traceback.print_exc()

# Test predict method
print("\nTesting predict method...")
try:
    context_data = df.iloc[:80]
    target_dates = pd.date_range(context_data['datetime'].iloc[-1], periods=13, freq='D')[1:]
    
    pred_df = predictor.predict(
        df=context_data[['open', 'high', 'low', 'close']],
        x_timestamp=context_data['datetime'],
        y_timestamp=target_dates,
        pred_len=12,
        verbose=True
    )
    
    print(f"\nPrediction successful!")
    print(f"Predictions shape: {pred_df.shape}")
    print(f"Predicted close prices:\n{pred_df['close']}")
    print(f"Any NaN? {pred_df['close'].isna().any()}")
    
except Exception as e:
    print(f"\n❌ Prediction failed with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

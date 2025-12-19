import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Binance API credentials
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Trading parameters
SYMBOL = "BTCUSDT"
TESTNET=True
INITIAL_BALANCE = 10000  # USD
FEE = 0.001  # 0.1% trading fee
TRADE_AMOUNT = 0.01  # BTC amount for micro-trades

# Training parameters
TOTAL_TIMESTEPS = 750000

# Model save path
MODEL_PATH = 'ppo_trading_model'

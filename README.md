# RL Crypto Trading Bot

A Reinforcement Learning (RL) crypto trading bot for Binance Spot market using PPO algorithm, designed to trade micro-changes.

## Features

- Custom Gymnasium environment for Binance order book data
- PPO agent using Stable Baselines3
- Real-time WebSocket data streaming from Binance
- Modular architecture for training and live trading
- Micro-trading focused on small position changes

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your Binance API keys:
   - Copy `.env` file and add your API key and secret
   - Get keys from [Binance API Management](https://www.binance.com/en/my/settings/api-management)

3. Ensure you have testnet API keys for safe testing (recommended)

## Usage

### Training Mode
Train the PPO agent on historical/simulated data:
```bash
python main.py train
```

### Live Trading Mode
Run the trained agent in live trading:
```bash
python main.py live
```

## Project Structure

- `env.py`: Custom Gymnasium trading environment
- `agent.py`: PPO agent implementation
- `data_stream.py`: Binance WebSocket data handler
- `config.py`: Configuration and parameters
- `main.py`: Entry point for training/live modes
- `requirements.txt`: Python dependencies
- `.env`: API keys (not committed to version control)

## Configuration

Edit `config.py` to adjust:
- Trading symbols
- Initial balance
- Trading fees
- Micro-trade amounts
- Training parameters

## Safety Notes

- Use testnet for initial testing
- Start with small amounts
- Monitor performance closely
- This is experimental software - use at your own risk

## Troubleshooting

- Ensure API keys are correctly set in `.env`
- Check internet connection for WebSocket data
- Verify Binance API permissions for trading
- For training issues, check Stable Baselines3 documentation
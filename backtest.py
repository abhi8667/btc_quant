import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import BinanceTradingEnv
from config import MODEL_PATH, INITIAL_BALANCE

# A simple wrapper to simulate the data stream for backtesting
class BacktestDataStream:
    def __init__(self, data):
        self.data = data
        self.idx = 0
    def get_order_book(self):
        row = self.data.iloc[self.idx]
        # Reconstruct the expected LOB format from CSV columns
        return {
            'bids': [[row[f'bid_p_{i}'], row[f'bid_v_{i}']] for i in range(20)],
            'asks': [[row[f'ask_p_{i}'], row[f'ask_v_{i}']] for i in range(20)]
        }

def run_test(test_csv):
    print(f"--- Running Visual Backtest on {test_csv} ---")
    df = pd.read_csv(test_csv)
    stream = BacktestDataStream(df)
    env = BinanceTradingEnv(stream, max_steps=len(df)-1)
    
    # Load the trained brain
    model = PPO.load(MODEL_PATH, env=env)
    
    obs, _ = env.reset()
    history = []

    for i in range(len(df)-1):
        stream.idx = i
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        history.append({
            'price': env.current_price,
            'action': action,
            'net_worth': info['net_worth']
        })
        if terminated or truncated: break

    # --- PLOTTING ---
    results = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Top: Price and Trade Dots
    ax1.plot(results['price'], label='Market Price', color='gray', alpha=0.4)
    buys = results[results['action'] == 1]
    sells = results[results['action'] == 2]
    ax1.scatter(buys.index, buys['price'], marker='^', color='green', label='BUY', s=80)
    ax1.scatter(sells.index, sells['price'], marker='v', color='red', label='SELL', s=80)
    ax1.set_title(f"Trading Decisions for {test_csv}")
    ax1.legend()

    # Bottom: Profit Curve
    ax2.plot(results['net_worth'], label='Bot Net Worth', color='blue', linewidth=2)
    ax2.axhline(y=INITIAL_BALANCE, color='black', linestyle='--', label='Start Balance')
    ax2.fill_between(results.index, INITIAL_BALANCE, results['net_worth'], 
                     where=(results['net_worth'] >= INITIAL_BALANCE), color='green', alpha=0.1)
    ax2.fill_between(results.index, INITIAL_BALANCE, results['net_worth'], 
                     where=(results['net_worth'] < INITIAL_BALANCE), color='red', alpha=0.1)
    ax2.set_title("Equity Growth (Net Worth)")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test("test_market_data.csv")
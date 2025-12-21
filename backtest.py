import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from stable_baselines3 import PPO
from env import BinanceTradingEnv
from data_stream import DataStream
from config import MODEL_PATH, INITIAL_BALANCE, SYMBOL
from binance.client import Client
from config import API_KEY, API_SECRET

def run_live_backtest(max_steps=1000):
    """
    Runs a backtest using the trained model with live market data.
    This simulates trading for a limited number of steps.
    """
    print(f"--- Running Live Backtest with Trained Model ---")
    print(f"Model: {MODEL_PATH}.zip")
    print(f"Symbol: {SYMBOL}")
    print(f"Max Steps: {max_steps}")
    print(f"Initial Balance: ${INITIAL_BALANCE:.2f}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"ERROR: Model file {MODEL_PATH}.zip not found. Please train the model first.")
        return
    
    # Initialize data stream
    client = Client(API_KEY, API_SECRET, testnet=True)
    data_stream = DataStream(client, SYMBOL)
    data_stream.start()
    
    # Wait for data to sync
    print("Syncing WebSocket with Order Book (approx. 5s)...")
    time.sleep(5)
    
    # Setup environment
    env = BinanceTradingEnv(data_stream, max_steps=max_steps, initial_balance=INITIAL_BALANCE)
    
    # Load the trained model
    print(f"Loading trained model from {MODEL_PATH}.zip...")
    model = PPO.load(MODEL_PATH, env=env)
    
    # Reset environment
    obs, _ = env.reset()
    history = []
    
    print("\nStarting backtest simulation...")
    print("-" * 80)
    
    try:
        for step in range(max_steps):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record history
            history.append({
                'step': step,
                'price': env.current_price,
                'action': action,
                'action_name': ['HOLD', 'BUY', 'SELL'][action],
                'net_worth': info['net_worth'],
                'balance': env.balance,
                'position': env.position,
                'reward': reward
            })
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step:4d} | Price: ${env.current_price:10.2f} | "
                      f"Action: {['HOLD', 'BUY', 'SELL'][action]:4s} | "
                      f"Net Worth: ${info['net_worth']:10.2f} | "
                      f"Reward: {reward:8.5f}")
            
            if terminated or truncated:
                print(f"\nEpisode ended at step {step}: {'Terminated' if terminated else 'Truncated'}")
                break
                
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
    finally:
        data_stream.stop()
        print("Data stream stopped.")
    
    # --- Analysis & Plotting ---
    if len(history) > 0:
        results = pd.DataFrame(history)
        
        # Calculate statistics
        final_net_worth = results['net_worth'].iloc[-1]
        total_return = ((final_net_worth - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        max_net_worth = results['net_worth'].max()
        min_net_worth = results['net_worth'].min()
        max_drawdown = ((max_net_worth - min_net_worth) / max_net_worth) * 100
        
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Initial Balance:     ${INITIAL_BALANCE:,.2f}")
        print(f"Final Net Worth:     ${final_net_worth:,.2f}")
        print(f"Total Return:        {total_return:+.2f}%")
        print(f"Max Net Worth:       ${max_net_worth:,.2f}")
        print(f"Min Net Worth:       ${min_net_worth:,.2f}")
        print(f"Max Drawdown:        {max_drawdown:.2f}%")
        print(f"Total Steps:         {len(results)}")
        print(f"Buy Actions:         {len(results[results['action'] == 1])}")
        print(f"Sell Actions:        {len(results[results['action'] == 2])}")
        print(f"Hold Actions:        {len(results[results['action'] == 0])}")
        print("=" * 80)
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Top: Price and Trade Dots
        ax1.plot(results['step'], results['price'], label='Market Price', color='gray', alpha=0.7, linewidth=1)
        buys = results[results['action'] == 1]
        sells = results[results['action'] == 2]
        if len(buys) > 0:
            ax1.scatter(buys['step'], buys['price'], marker='^', color='green', label='BUY', s=100, alpha=0.7, zorder=5)
        if len(sells) > 0:
            ax1.scatter(sells['step'], sells['price'], marker='v', color='red', label='SELL', s=100, alpha=0.7, zorder=5)
        ax1.set_ylabel('Price (USD)')
        ax1.set_title(f"Trading Decisions - {SYMBOL} (Total Return: {total_return:+.2f}%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Middle: Profit Curve
        ax2.plot(results['step'], results['net_worth'], label='Bot Net Worth', color='blue', linewidth=2)
        ax2.axhline(y=INITIAL_BALANCE, color='black', linestyle='--', label='Start Balance', linewidth=1)
        ax2.fill_between(results['step'], INITIAL_BALANCE, results['net_worth'], 
                         where=(results['net_worth'] >= INITIAL_BALANCE), color='green', alpha=0.2)
        ax2.fill_between(results['step'], INITIAL_BALANCE, results['net_worth'], 
                         where=(results['net_worth'] < INITIAL_BALANCE), color='red', alpha=0.2)
        ax2.set_ylabel('Net Worth (USD)')
        ax2.set_title("Equity Growth (Net Worth)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Bottom: Rewards
        ax3.plot(results['step'], results['reward'], label='Reward', color='purple', alpha=0.7, linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Reward')
        ax3.set_title("Reward Over Time")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: backtest_results.png")
        plt.show()
    else:
        print("No data collected during backtest.")

if __name__ == "__main__":
    import sys
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    run_live_backtest(max_steps=max_steps)

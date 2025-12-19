import sys
import time
import os
from binance.client import Client  # Corrected import
from config import API_KEY, API_SECRET, SYMBOL, TOTAL_TIMESTEPS, MODEL_PATH
from data_stream import DataStream
from env import BinanceTradingEnv
from agent import TradingAgent

def main():
    # 1. Validation & Mode Selection
    if len(sys.argv) < 2:
        print("Usage: python main.py <mode>")
        print("Modes: train, live")
        return
    
    mode = sys.argv[1].lower()
    
    # 2. Initialize Hardware & Streams
    print(f"--- Launching Continuum-RL in {mode.upper()} mode ---")
    
    # Initialize the Binance Client with the Testnet flag
    # In 2025, always ensure testnet=True for safety during development
    client = Client(API_KEY, API_SECRET, testnet=True)
    
    # Pass the pre-configured client to your DataStream
    # IMPORTANT: Ensure your data_stream.py __init__ accepts (client, symbol)
    data_stream = DataStream(client, SYMBOL)
    data_stream.start()
    
    # Allow buffer to sync with Binance Order Book
    print("Syncing WebSocket with Order Book (approx. 5s)...")
    time.sleep(5)

    # 3. Setup Environment & Agent
    env = BinanceTradingEnv(data_stream)
    agent = TradingAgent(env)

    # 4. Mode Logic
    if mode == 'train':
        print("Starting Deep RL Training Session...")
        
        # RESUME LOGIC: Picks up where the brain left off
        if os.path.exists(MODEL_PATH + ".zip"):
            print(f"Loading existing model from {MODEL_PATH} to resume training...")
            agent.load_model()
        else:
            print("No existing model found. Starting fresh session.")

        try:
            agent.train(total_timesteps=TOTAL_TIMESTEPS)
            agent.save_model()
            print(f"Successfully trained and saved model to {MODEL_PATH}")
        except Exception as e:
            print(f"Training failed: {e}")
        finally:
            data_stream.stop()

    elif mode == 'live':
        print("Starting Live Trading with Incremental Learning...")
        
        # Load the trained model
        if os.path.exists(MODEL_PATH + ".zip"):
            agent.load_model()
            print("Pre-trained model loaded.")
        else:
            print("CRITICAL ERROR: No model found. Run 'train' mode first.")
            data_stream.stop()
            return

        obs, _ = env.reset() # Gymnasium returns (obs, info)
        step_count = 0
        
        try:
            while True:
                # Agent makes a deterministic decision for live trading
                action = agent.predict(obs)
                
                # Execute in our virtualized Binance Env (Unpacking 5 Gymnasium values)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Log Status
                net_worth = info.get('net_worth', 0)
                print(f"[Step {step_count}] Action: {action} | Reward: {reward:.5f} | Net Worth: ${net_worth:.2f}")

                # --- ONLINE ADAPTATION ---
                # Updates the brain every 500 steps without resetting its knowledge
                step_count += 1
                if step_count % 500 == 0:
                    print(">>> REFINING MODEL: Adapting to latest market regime...")
                    agent.update_live(timesteps=100)
                    agent.save_model() 

                # --- SAFETY: KILL SWITCH ---
                if net_worth < (env.initial_balance * 0.85):
                    print("CRITICAL: 15% Max Drawdown reached. Safety Kill Switch triggered.")
                    break
                
                if done:
                    obs, _ = env.reset()

                # Optimized for micro-change frequency
                time.sleep(0.5) 

        except KeyboardInterrupt:
            print("\nEmergency Stop requested by user.")
        finally:
            print("Finalizing brain state and shutting down...")
            agent.save_model()
            data_stream.stop()
            print("System offline.")

if __name__ == '__main__':
    main()

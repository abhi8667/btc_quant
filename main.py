import sys
import time
import os
from binance.client import Client 
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
    
    # 2. Initialize Binance Connection
    print(f"--- Launching Continuum-RL in {mode.upper()} mode ---")
    
    # Create the Client first to manage API settings and Testnet status
    client = Client(API_KEY, API_SECRET, testnet=True)
    
    # Initialize the data stream with the client and symbol
    data_stream = DataStream(client, SYMBOL)
    
    try:
        # Start the WebSocket streams (includes initial snapshot fetch and synchronization)
        data_stream.start()
        
        # Verify data stream is ready before proceeding
        if not data_stream.is_ready():
            raise RuntimeError("Data stream is not ready after start(). Check connection and symbol.")
        
        # 3. Setup Environment & Agent
        # The environment will verify data is ready in its reset() method
        env = BinanceTradingEnv(data_stream)
        agent = TradingAgent(env)

        # 4. Mode Logic
        if mode == 'train':
            print("Starting Deep RL Training Session...")
            
            # RESUME LOGIC: Continue from checkpoint if it exists and is compatible
            if os.path.exists(MODEL_PATH + ".zip"):
                print(f"Attempting to load existing model from {MODEL_PATH}...")
                agent.load_model()  # This will handle observation space mismatches gracefully
            else:
                print("No existing model found. Starting a fresh training session...")

            # Train the agent
            agent.train(total_timesteps=TOTAL_TIMESTEPS)
            agent.save_model()
            print(f"Training session complete. Final model saved to {MODEL_PATH}")

        elif mode == 'live':
            print("Starting Live Trading with Incremental Learning...")
            
            # Load your best trained model
            if os.path.exists(MODEL_PATH + ".zip"):
                agent.load_model()
                print("Pre-trained model successfully loaded.")
            else:
                print("CRITICAL: No trained model found. Please run 'train' mode first.")
                return

            # Gymnasium 0.26+ reset returns (observation, info)
            obs, info = env.reset()
            step_count = 0
            
            while True:
                # Agent makes a decision based on live LOB data
                action = agent.predict(obs)
                
                # Execute action and handle the 5 Gymnasium return values
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Log performance
                net_worth = info.get('net_worth', 0)
                print(f"[Step {step_count}] Action: {action} | Reward: {reward:.5f} | Net Worth: ${net_worth:.2f}")

                # --- ONLINE REFINEMENT ---
                # Every 500 steps, let the bot "fine-tune" its brain on recent market data
                step_count += 1
                if step_count % 500 == 0:
                    print(">>> Market Adaptation: Performing incremental update...")
                    agent.update_live(timesteps=100)
                    agent.save_model() 

                # Safety Check: Stop if drawdown is too deep
                if net_worth < (env.initial_balance * 0.80):
                    print("SAFETY ALERT: Maximum 20% Drawdown reached. Emergency halt.")
                    break
                
                if done:
                    obs, _ = env.reset()

                # Optimized loop interval to keep pace with Binance tick updates
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nShutdown signal received (Ctrl+C).")
    except Exception as e:
        print(f"CRITICAL SYSTEM FAILURE: {e}")
    finally:
        # 5. CLEANUP
        # This prevents the 'cannot schedule new futures after shutdown' error
        print("Shutting down background threads and closing WebSocket connection...")
        data_stream.stop()
        print("Continuum-RL Offline.")

if __name__ == '__main__':
    main()

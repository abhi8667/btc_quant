import sys
import time
from config import API_KEY, API_SECRET, SYMBOL, TOTAL_TIMESTEPS, MODEL_PATH
from data_stream import DataStream
from env import BinanceTradingEnv
from agent import TradingAgent

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <mode>")
        print("Modes: train, live")
        return
    
    mode = sys.argv[1].lower()
    
    # Initialize data stream
    print("Initializing data stream...")
    data_stream = DataStream(API_KEY, API_SECRET, SYMBOL)
    
    # Wait for WebSocket connection to establish
    time.sleep(5)
    
    # Initialize environment
    print("Initializing trading environment...")
    env = BinanceTradingEnv(data_stream)
    
    # Initialize agent
    print("Initializing PPO agent...")
    agent = TradingAgent(env)
    
    if mode == 'train':
        print("Starting training mode...")
        agent.train(total_timesteps=TOTAL_TIMESTEPS)
        agent.save_model()
        print(f"Training completed. Model saved to {MODEL_PATH}")
    
    elif mode == 'live':
        print("Starting live trading mode...")
        # Load trained model
        try:
            agent.load_model()
            print("Model loaded successfully")
        except:
            print("No trained model found. Please train first.")
            return
        
        # Reset environment
        obs, _ = env.reset()
        
        print("Starting live trading...")
        try:
            while True:
                # Get action from agent
                action = agent.predict(obs)
                
                # Execute action in environment
                obs, reward, done, _, _ = env.step(action)
                
                # Log current state
                print(f"Action: {action}, Reward: {reward:.4f}, Balance: {env.balance:.2f}, Position: {env.position:.4f}")
                
                # Small delay to prevent overwhelming the API
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Live trading stopped by user")
        finally:
            data_stream.close()
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: train, live")

if __name__ == '__main__':
    main()
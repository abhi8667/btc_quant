import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from config import MODEL_PATH

class TradingAgent:
    def __init__(self, env):
        """
        Initializes the PPO agent with tuned hyperparameters for 
        stable learning in volatile crypto markets.
        """
        self.env = env
        self.log_dir = "./ppo_trading_logs/"
        
        # High-end tuned PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            tensorboard_log=self.log_dir,
            
            # --- HYPERPARAMETER TUNING ---
            learning_rate=0.0001,      # Lower rate reduces wild strategy shifts
            n_steps=2048,              # Steps per environment update
            batch_size=128,            # Larger batch for more stable updates
            n_epochs=10,               # Optimization passes per update
            gamma=0.99,                # Discount factor for future rewards
            gae_lambda=0.95,           # Bias vs variance trade-off
            clip_range=0.2,            # Limits policy update magnitude
            ent_coef=0.01,             # Balances exploration vs exploitation
            
            device="auto"              # Auto-detects GPU (CUDA)
        )

    def train(self, total_timesteps):
        """Trains the agent with periodic checkpoints."""
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path="./checkpoints/",
            name_prefix="continuum_rl"
        )
        
        print(f"Starting training for {total_timesteps} steps...")
        self.model.learn(
            total_timesteps=total_timesteps, 
            callback=checkpoint_callback,
            tb_log_name="train_run"
        )

    def save_model(self):
        """Saves model state."""
        self.model.save(MODEL_PATH)

    def load_model(self):
        """Loads a pre-trained agent."""
        self.model = PPO.load(MODEL_PATH, env=self.env)

    def predict(self, obs):
        """Returns deterministic actions for live trading."""
        action, _states = self.model.predict(obs, deterministic=True)
        return action

    def update_live(self, timesteps=100):
        """Performs Online Learning to adapt to new trends."""
        self.model.learn(
            total_timesteps=timesteps, 
            reset_num_timesteps=False, 
            tb_log_name="live_update"
        )

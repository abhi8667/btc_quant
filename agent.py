import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from config import MODEL_PATH

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    Decreases the learning rate from the initial_value to 0 as training progresses.
    This helps the AI 'settle' into a strategy rather than making wild guesses late in the run.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining starts at 1.0 and drops to 0.0
        return progress_remaining * initial_value
    return func

class TradingAgent:
    def __init__(self, env):
        """
        Initializes the PPO agent with a dynamic learning rate and 
        tuned hyperparameters for stable crypto trading.
        """
        self.env = env
        self.log_dir = "./ppo_trading_logs/"
        
        # High-end tuned PPO model
        self.model = PPO(
            policy="MlpPolicy", # Multi-Layer Perceptron for our 85-feature vector
            env=self.env,
            verbose=1,
            tensorboard_log=self.log_dir,
            
            # --- HYPERPARAMETER TUNING ---
            # Using the schedule: starts at 0.0003 and decays to 0
            learning_rate=linear_schedule(0.0003), 
            
            n_steps=2048,              # Steps per environment update
            batch_size=128,            # Larger batch for more stable updates
            n_epochs=10,               # Optimization passes per update
            gamma=0.99,                # Discount factor for future rewards
            gae_lambda=0.95,           # Bias vs variance trade-off
            clip_range=0.2,            # Limits policy update magnitude
            ent_coef=0.01,             # Balances exploration vs exploitation
            
            device="auto"              # Auto-detects GPU (CUDA) if available
        )

    def train(self, total_timesteps):
        """Trains the agent with periodic checkpoints to prevent data loss."""
        # Automatically saves the model every 50,000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=50000, 
            save_path="./checkpoints/",
            name_prefix="continuum_rl"
        )
        
        print(f"Starting training for {total_timesteps} steps with Linear Decay...")
        self.model.learn(
            total_timesteps=total_timesteps, 
            callback=checkpoint_callback,
            tb_log_name="train_run",
            reset_num_timesteps=False # Keeps the global step count consistent
        )

    def save_model(self):
        """Saves model state to the path specified in config.py."""
        self.model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def load_model(self):
        """Loads a pre-trained agent and links it to the current environment."""
        if os.path.exists(MODEL_PATH + ".zip"):
            try:
                # Try to load the model
                self.model = PPO.load(MODEL_PATH, env=self.env)
                print("Successfully loaded pre-trained model.")
            except (ValueError, RuntimeError, AssertionError) as e:
                # Handle observation space mismatch or other compatibility issues
                error_msg = str(e).lower()
                if "observation space" in error_msg or "observation spaces" in error_msg or "do not match" in error_msg:
                    print(f"WARNING: Existing model has incompatible observation space.")
                    print(f"The saved model was trained with a different feature count.")
                    print(f"Starting fresh training with current environment configuration...")
                    # Re-initialize the model with the current environment
                    self.model = PPO(
                        policy="MlpPolicy",
                        env=self.env,
                        verbose=1,
                        tensorboard_log=self.log_dir,
                        learning_rate=linear_schedule(0.0003),
                        n_steps=2048,
                        batch_size=128,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        ent_coef=0.01,
                        device="auto"
                    )
                else:
                    # Re-raise if it's a different error we don't know how to handle
                    print(f"Error loading model: {e}")
                    raise
        else:
            print("No model found at specified path. Training from scratch.")

    def predict(self, obs):
        """Returns deterministic actions (no randomness) for live trading."""
        action, _states = self.model.predict(obs, deterministic=True)
        return action

    def update_live(self, timesteps=100):
        """
        Performs Online Learning to adapt to new trends.
        Uses a fixed small learning rate for live adaptation.
        """
        self.model.learn(
            total_timesteps=timesteps, 
            reset_num_timesteps=False, 
            tb_log_name="live_update"
        )

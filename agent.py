from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from config import MODEL_PATH

class TradingAgent:
    def __init__(self, env):
        # Wrap environment for vectorized training
        self.env = DummyVecEnv([lambda: env])
        # Initialize PPO model with MLP policy
        self.model = PPO('MlpPolicy', self.env, verbose=1)
    
    def train(self, total_timesteps=10000):
        """Train the agent for the specified number of timesteps"""
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, observation):
        """Predict action for given observation"""
        action, _states = self.model.predict(observation)
        return action
    
    def save_model(self, path=MODEL_PATH):
        """Save the trained model"""
        self.model.save(path)
    
    def load_model(self, path=MODEL_PATH):
        """Load a trained model"""
        self.model = PPO.load(path, env=self.env)
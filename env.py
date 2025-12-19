import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import time

class BinanceTradingEnv(gym.Env):
    def __init__(self, data_stream, max_steps=1000, initial_balance=1000):
        super(BinanceTradingEnv, self).__init__()
        self.data_stream = data_stream
        self.max_steps = max_steps
        self.initial_balance = initial_balance
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # 83 Features: [40 LOB values] + [OBI, VWAP_Diff, RSI]
        self.observation_space = spaces.Box(low=-5, high=5, shape=(83,), dtype=np.float32)
        
        self.FEE = 0.00075  # 0.075% (Binance BNB fee)

    def reset(self, seed=None, options=None):
        """Resets the environment and returns (observation, info)."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0.0 
        self.current_price = 0.0
        self.step_count = 0
        self.cost_basis = 0.0
        self.steps_held = 0
        self.net_worth = self.initial_balance
        
        self.price_history = deque(maxlen=15)
        self.vwap_cum_price_vol = 0.0
        self.vwap_cum_vol = 0.0
        
        # --- FIX: WAIT FOR REAL-TIME DATA ---
        # Prevents ZeroDivisionError by ensuring WebSocket data has arrived
        print("Waiting for initial market price from WebSocket...")
        while self.current_price == 0:
            order_book = self.data_stream.get_order_book()
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if bids and asks:
                self.current_price = (float(bids[0][0]) + float(asks[0][0])) / 2
            else:
                time.sleep(0.5) # Wait for stream to populate

        # Return (obs, info) as required by Gymnasium
        return self._get_observation(), {"net_worth": self.net_worth}

    def step(self, action):
        """Executes action and returns (obs, reward, terminated, truncated, info)."""
        self.step_count += 1
        prev_net_worth = self.net_worth
        
        # 1. Execute Actions
        if action == 1:  # BUY
            amount_to_buy = self.balance / (self.current_price * (1 + self.FEE))
            if amount_to_buy > 0:
                self.position += amount_to_buy
                self.balance = 0
                self.cost_basis = self.current_price
                self.steps_held = 0
        
        elif action == 2: # SELL
            if self.position > 0:
                self.balance = self.position * self.current_price * (1 - self.FEE)
                self.position = 0
                self.cost_basis = 0
                self.steps_held = 0

        # 2. Update Environment State
        if self.position > 0:
            self.steps_held += 1
            
        obs = self._get_observation()
        self.net_worth = self.balance + (self.position * self.current_price)
        
        # 3. HIGH-END REWARD LOGIC
        # Prevents division by zero if net worth drops to zero
        step_return = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-9)
        holding_penalty = 0.0001 * self.steps_held if self.position > 0 else 0
        reward = step_return - holding_penalty
        
        # 4. TERMINATION vs TRUNCATION
        terminated = self.step_count >= self.max_steps # Natural end
        truncated = self.net_worth < (self.initial_balance * 0.7) # Safety end (30% loss)
        
        # Return 5 values
        return obs, float(reward), terminated, truncated, {"net_worth": self.net_worth}

    def _get_observation(self):
        """Constructs the stationarized observation vector."""
        order_book = self.data_stream.get_order_book()
        bids = order_book.get('bids', [])[:20]
        asks = order_book.get('asks', [])[:20]
        
        # Safe Mid-Price Update
        if bids and asks:
            self.current_price = (float(bids[0][0]) + float(asks[0][0])) / 2
        
        # Padding for LOB consistency
        while len(bids) < 20: bids.append([bids[-1][0] if bids else self.current_price, 0])
        while len(asks) < 20: asks.append([asks[-1][0] if asks else self.current_price, 0])
        
        # --- STATIONARY FEATURE ENGINEERING ---
        # Normalize Prices relative to current Mid Price
        normalized_lob = []
        for p, v in bids:
            normalized_lob.extend([(float(p) / (self.current_price + 1e-9)) - 1, np.log1p(float(v))])
        for p, v in asks:
            normalized_lob.extend([(float(p) / (self.current_price + 1e-9)) - 1, np.log1p(float(v))])
            
        # Technical Indicators
        self.price_history.append(self.current_price)
        rsi = self._calculate_rsi() / 100.0
        
        bid_vol = sum(float(v) for _, v in bids)
        ask_vol = sum(float(v) for _, v in asks)
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        
        self.vwap_cum_price_vol += self.current_price
        self.vwap_cum_vol += 1
        vwap = self.vwap_cum_price_vol / self.vwap_cum_vol
        vwap_diff = (self.current_price / (vwap + 1e-9)) - 1
        
        full_obs = np.array(normalized_lob + [obi, vwap_diff, rsi], dtype=np.float32)
        return np.clip(full_obs, -5, 5)

    def _calculate_rsi(self, window=14):
        if len(self.price_history) < window: return 50.0
        prices = np.array(self.price_history)
        deltas = np.diff(prices)
        up = deltas[deltas > 0].sum() / window
        down = -deltas[deltas < 0].sum() / window
        if down == 0: return 100.0
        rs = up / (down + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

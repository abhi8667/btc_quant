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
        
        # Updated Feature Count: 85
        # [40 LOB prices/vols] + [OBI, VWAP_Diff, RSI, EMA_Fast_Diff, EMA_Slow_Diff]
        self.observation_space = spaces.Box(low=-5, high=5, shape=(85,), dtype=np.float32)
        
        self.FEE = 0.00075  # 0.075% (Standard Binance fee)

    def reset(self, seed=None, options=None):
        """Resets the environment and returns (observation, info)."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0.0 
        self.current_price = 0.0
        self.cost_basis = 0.0  # Tracks entry price for net profit calculation
        self.step_count = 0
        self.steps_held = 0
        self.net_worth = self.initial_balance
        
        # Expanded history for better indicator calculation (EMA/RSI)
        self.price_history = deque(maxlen=100)
        self.vwap_cum_price_vol = 0.0
        self.vwap_cum_vol = 0.0
        
        # Synchronization loop to prevent ZeroDivisionError
        print("Synchronizing with live WebSocket data...")
        start_wait = time.time()
        while self.current_price == 0:
            if time.time() - start_wait > 30:
                raise TimeoutError("WebSocket failed to provide data. Check network.")
            
            order_book = self.data_stream.get_order_book()
            bids, asks = order_book.get('bids', []), order_book.get('asks', [])
            
            if bids and asks:
                self.current_price = (float(bids[0][0]) + float(asks[0][0])) / 2
                print(f"Environment synchronized. Initial Price: {self.current_price}")
            else:
                time.sleep(1)

        return self._get_observation(), {"net_worth": self.net_worth}

    def step(self, action):
        """Executes action and returns (obs, reward, terminated, truncated, info)."""
        self.step_count += 1
        prev_net_worth = self.net_worth
        realized_reward = 0.0  # Bonus awarded only when closing profitable trades
        
        # 1. Execute Actions
        if action == 1 and self.balance > 10:  # BUY
            self.cost_basis = self.current_price
            amount_to_buy = self.balance / (self.current_price * (1 + self.FEE) + 1e-9)
            self.position += amount_to_buy
            self.balance = 0
            self.steps_held = 0
        
        elif action == 2 and self.position > 0: # SELL
            # Net Realized Profit Calculation (must beat round-trip fees ~0.15%)
            net_exit_val = self.current_price * (1 - self.FEE)
            net_entry_cost = self.cost_basis * (1 + self.FEE)
            net_profit_pct = (net_exit_val / (net_entry_cost + 1e-9)) - 1
            
            if net_profit_pct > 0:
                # Big Reward for actual net profit
                realized_reward = net_profit_pct * 50.0 
                print(f"TRADE SUCCESS: Net profit of {net_profit_pct:.4%}")
            else:
                # Penalty for closing a trade at a loss
                realized_reward = net_profit_pct * 5.0

            self.balance = self.position * self.current_price * (1 - self.FEE)
            self.position = 0
            self.cost_basis = 0
            self.steps_held = 0

        # 2. Update Internal State
        if self.position > 0: self.steps_held += 1
        obs = self._get_observation()
        self.net_worth = self.balance + (self.position * self.current_price)
        
        # 3. COMBINED REWARD LOGIC
        # Unrealized PnL guides the agent; realized_reward provides the "Win" signal
        unrealized_return = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-9)
        idle_penalty = 0.0001 * self.steps_held if self.position > 0 else 0
        
        reward = unrealized_return + realized_reward - idle_penalty
        
        # 4. Termination Logic
        terminated = self.step_count >= self.max_steps
        truncated = self.net_worth < (self.initial_balance * 0.7) # Stop at 30% loss
        
        return obs, float(reward), terminated, truncated, {"net_worth": self.net_worth}

    def _get_observation(self):
        """Constructs the stationarized observation vector with indicators."""
        order_book = self.data_stream.get_order_book()
        bids, asks = order_book.get('bids', [])[:20], order_book.get('asks', [])[:20]
        
        if bids and asks:
            self.current_price = (float(bids[0][0]) + float(asks[0][0])) / 2
        
        # Maintain price history for indicators
        self.price_history.append(self.current_price)
        prices = np.array(self.price_history)

        # Padding LOB data
        while len(bids) < 20: bids.append([bids[-1][0] if bids else self.current_price, 0])
        while len(asks) < 20: asks.append([asks[-1][0] if asks else self.current_price, 0])
        
        # Normalize LOB (Price difference from mid-price)
        normalized_lob = []
        for p, v in bids:
            normalized_lob.extend([(float(p) / (self.current_price + 1e-9)) - 1, np.log1p(float(v))])
        for p, v in asks:
            normalized_lob.extend([(float(p) / (self.current_price + 1e-9)) - 1, np.log1p(float(v))])
            
        # Indicator 1: RSI (Scaled 0 to 1)
        rsi = self._calculate_rsi(prices, window=14) / 100.0
        
        # Indicator 2: Order Book Imbalance (OBI)
        bid_vol, ask_vol = sum(float(v) for _, v in bids), sum(float(v) for _, v in asks)
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        
        # Indicator 3: VWAP Difference
        self.vwap_cum_price_vol += self.current_price
        self.vwap_cum_vol += 1
        vwap = self.vwap_cum_price_vol / self.vwap_cum_vol
        vwap_diff = (self.current_price / (vwap + 1e-9)) - 1

        # Indicator 4 & 5: EMA Differences (Trend detection)
        ema_fast = self._calculate_ema(prices, span=12)
        ema_slow = self._calculate_ema(prices, span=26)
        ema_fast_diff = (self.current_price / (ema_fast + 1e-9)) - 1
        ema_slow_diff = (self.current_price / (ema_slow + 1e-9)) - 1
        
        # Combine all 85 features
        full_obs = np.array(
            normalized_lob + [obi, vwap_diff, rsi, ema_fast_diff, ema_slow_diff], 
            dtype=np.float32
        )
        return np.clip(full_obs, -5, 5)

    def _calculate_ema(self, prices, span):
        if len(prices) < 2: return self.current_price
        alpha = 2 / (span + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * alpha) + (ema * (1 - alpha))
        return ema

    def _calculate_rsi(self, prices, window=14):
        if len(prices) < window: return 50.0
        deltas = np.diff(prices)
        up = deltas[deltas > 0].sum() / window
        down = -deltas[deltas < 0].sum() / window
        if down == 0: return 100.0
        rs = up / (down + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from config import INITIAL_BALANCE, TRADE_AMOUNT

class BinanceTradingEnv(gym.Env):
    def __init__(self, data_stream, max_steps=1000):
        super(BinanceTradingEnv, self).__init__()
        self.data_stream = data_stream
        self.max_steps = max_steps
        
        # Action space: 0 = Hold, 1 = Buy/Long, 2 = Sell/Exit
        self.action_space = spaces.Discrete(3)
        
        # Observation space: order book (80) + OBI (1) + price_vs_vwap (1) + RSI (1) = 83
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(83,), dtype=np.float32)
        
        # Fees: 0.1% taker fee
        self.FEE = 0.001
        
        # Initialize state
        self.balance = INITIAL_BALANCE
        self.position = 0.0  # BTC held
        self.current_price = 0.0
        self.step_count = 0
        self.total_profit = 0.0
        self.cost_basis = 0.0  # Average cost per BTC
        self.steps_held = 0
        
        # For indicators
        self.price_history = deque(maxlen=14)  # For RSI
        self.vwap_cum_price_vol = 0.0
        self.vwap_cum_vol = 0.0
        self.prev_rsi = 50.0  # Initial RSI
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset environment state
        self.balance = INITIAL_BALANCE
        self.position = 0.0
        self.current_price = 0.0
        self.step_count = 0
        self.total_profit = 0.0
        self.cost_basis = 0.0
        self.steps_held = 0
        
        # Reset indicators
        self.price_history.clear()
        self.vwap_cum_price_vol = 0.0
        self.vwap_cum_vol = 0.0
        self.prev_rsi = 50.0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        
        # Update holding time
        if self.position > 0:
            self.steps_held += 1
        
        # Execute action
        reward = 0.0
        if action == 1:  # Buy/Long
            cost = TRADE_AMOUNT * self.current_price * (1 + self.FEE)
            if self.balance >= cost:
                # Update cost basis
                total_cost = self.position * self.cost_basis + cost
                self.position += TRADE_AMOUNT
                self.cost_basis = total_cost / self.position
                self.balance -= cost
        elif action == 2:  # Sell/Exit
            if self.position >= TRADE_AMOUNT:
                revenue = TRADE_AMOUNT * self.current_price * (1 - self.FEE)
                # Calculate profit for this trade
                trade_profit = (self.current_price - self.cost_basis) * TRADE_AMOUNT - (self.current_price * TRADE_AMOUNT * self.FEE * 2)  # Fee on buy and sell
                self.total_profit += trade_profit
                self.balance += revenue
                self.position -= TRADE_AMOUNT
                if self.position == 0:
                    self.cost_basis = 0.0
                    self.steps_held = 0
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward: Profit after fees - Holding Time Penalty
        holding_penalty = 0.001 * self.steps_held if self.position > 0 else 0.0
        reward = self.total_profit - holding_penalty
        
        # Check if done
        done = self.step_count >= self.max_steps or self.balance <= 0
        
        return obs, reward, done, False, {}
    
    def _get_observation(self):
        # Get current order book
        order_book = self.data_stream.get_order_book()
        
        # Extract top 20 bids and asks
        bids = order_book.get('bids', [])[:20]
        asks = order_book.get('asks', [])[:20]
        
        # Pad if necessary
        while len(bids) < 20:
            bids.append([0.0, 0.0])
        while len(asks) < 20:
            asks.append([0.0, 0.0])
        
        # Update current price (mid price)
        if bids and asks:
            self.current_price = (float(bids[0][0]) + float(asks[0][0])) / 2
        
        # Update price history for RSI
        self.price_history.append(self.current_price)
        
        # Update VWAP (assume volume = TRADE_AMOUNT per step for simplicity)
        volume = TRADE_AMOUNT
        self.vwap_cum_price_vol += self.current_price * volume
        self.vwap_cum_vol += volume
        vwap = self.vwap_cum_price_vol / self.vwap_cum_vol if self.vwap_cum_vol > 0 else self.current_price
        
        # Calculate OBI (Order Book Imbalance)
        bid_vol = sum(float(vol) for _, vol in bids)
        ask_vol = sum(float(vol) for _, vol in asks)
        obi = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.0
        
        # Calculate RSI
        rsi = self._calculate_rsi()
        
        # Price vs VWAP
        price_vs_vwap = self.current_price - vwap
        
        # Create observation array
        obs = []
        for price, volume in bids:
            obs.extend([float(price), float(volume)])
        for price, volume in asks:
            obs.extend([float(price), float(volume)])
        obs.extend([obi, price_vs_vwap, rsi])
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_rsi(self):
        if len(self.price_history) < 2:
            return 50.0
        
        gains = []
        losses = []
        for i in range(1, len(self.price_history)):
            change = self.price_history[i] - self.price_history[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        if avg_loss == 0:
            rs = 100
        else:
            rs = avg_gain / avg_loss
        
        rsi = 100 - (100 / (1 + rs))
        return rsi
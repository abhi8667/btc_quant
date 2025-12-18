from binance.client import Client
from binance.websockets import BinanceSocketManager
import time
from collections import deque
import threading

class DataStream:
    def __init__(self, api_key, api_secret, symbol):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.order_book = {'bids': [], 'asks': []}
        self.tick_buffer = deque(maxlen=6000)  # Buffer for last ~60 seconds of tick data (assuming ~100 ticks/sec max)
        self.bm = BinanceSocketManager(self.client)
        self.conn_key_depth = None
        self.conn_key_trade = None
        self.running = True
        
        # Start WebSocket connections
        self._start_connections()
        
        # Start a thread for automatic reconnect
        self.reconnect_thread = threading.Thread(target=self._reconnect_loop)
        self.reconnect_thread.daemon = True
        self.reconnect_thread.start()

    def _start_connections(self):
        try:
            # Start depth socket for order book data (top 20 levels)
            self.conn_key_depth = self.bm.start_depth_socket(
                self.symbol, 
                self._process_message, 
                depth=20
            )
            # Start trade socket for tick data
            self.conn_key_trade = self.bm.start_trade_socket(
                self.symbol,
                self._process_message
            )
            self.bm.start()
            print("WebSocket connections started successfully")
        except Exception as e:
            print(f"Error starting connections: {e}")
            self._reconnect()

    def _process_message(self, msg):
        try:
            if msg['e'] == 'depthUpdate':
                # Update order book
                self.order_book['bids'] = msg['b']
                self.order_book['asks'] = msg['a']
            elif msg['e'] == 'trade':
                # Store tick data
                tick = {
                    'event_time': msg['E'],
                    'trade_time': msg['T'],
                    'symbol': msg['s'],
                    'price': float(msg['p']),
                    'quantity': float(msg['q']),
                    'buyer_order_id': msg['b'],
                    'seller_order_id': msg['a'],
                    'trade_id': msg['t'],
                    'is_buyer_maker': msg['m']
                }
                self.tick_buffer.append(tick)
        except Exception as e:
            print(f"Error processing message: {e}")

    def _reconnect_loop(self):
        while self.running:
            time.sleep(10)  # Check every 10 seconds
            if not self._is_connected():
                print("Connection lost, attempting to reconnect...")
                self._reconnect()

    def _is_connected(self):
        # Simple check - in a real implementation, you might ping or check socket status
        return self.conn_key_depth is not None and self.conn_key_trade is not None

    def _reconnect(self):
        try:
            self.close()
            time.sleep(5)  # Wait before reconnecting
            self._start_connections()
        except Exception as e:
            print(f"Reconnect failed: {e}")

    def get_order_book(self):
        return self.order_book

    def get_tick_buffer(self, seconds=60):
        """Get tick data from the last specified seconds"""
        current_time = time.time() * 1000  # Convert to milliseconds
        cutoff_time = current_time - (seconds * 1000)
        return [tick for tick in self.tick_buffer if tick['event_time'] > cutoff_time]

    def convert_to_observation(self, order_book=None):
        """Convert raw order book data to observation format for RL model"""
        import numpy as np
        
        if order_book is None:
            order_book = self.get_order_book()
        
        # Extract top 20 bids and asks
        bids = order_book.get('bids', [])[:20]
        asks = order_book.get('asks', [])[:20]
        
        # Pad if necessary
        while len(bids) < 20:
            bids.append([0.0, 0.0])
        while len(asks) < 20:
            asks.append([0.0, 0.0])
        
        # Create observation array (order book part only - indicators calculated in env)
        obs = []
        for price, volume in bids:
            obs.extend([float(price), float(volume)])
        for price, volume in asks:
            obs.extend([float(price), float(volume)])
        
        return np.array(obs, dtype=np.float32)

    def close(self):
        try:
            if self.conn_key_depth:
                self.bm.stop_socket(self.conn_key_depth)
            if self.conn_key_trade:
                self.bm.stop_socket(self.conn_key_trade)
            self.bm.close()
        except Exception as e:
            print(f"Error closing connections: {e}")
        finally:
            self.conn_key_depth = None
            self.conn_key_trade = None
            self.running = False
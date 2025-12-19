import time
from binance import ThreadedWebsocketManager
from collections import deque
import numpy as np

class DataStream:
    def __init__(self, client, symbol):
        """
        Initializes the data stream using a pre-configured Binance Client.
        The client object already contains the API Key, Secret, and Testnet settings.
        """
        self.symbol = symbol.lower()
        
        # Initialize the ThreadedWebsocketManager using credentials from the client
        # Accessing the api_key and api_secret stored within the Client instance
        self.twm = ThreadedWebsocketManager(
            api_key=client.API_KEY, 
            api_secret=client.API_SECRET,
            testnet=client.testnet  # Ensures the stream matches the testnet setting
        )
        
        self.order_book = {'bids': {}, 'asks': {}} # Dict for O(1) depth updates
        self.tick_buffer = deque(maxlen=1000)
        self.last_update_id = 0

    def start(self):
        """Starts the internal loop and initiates socket connections."""
        self.twm.start() # Start is required to initialize the internal thread
        
        # Start Partial Book Depth Stream (Faster for RL than full diff depth)
        self.twm.start_depth_socket(
            callback=self._handle_depth_message, 
            symbol=self.symbol,
            depth=20  # Matches the observation space in your environment
        )
        
        # Start Trade Stream for micro-tick data
        self.twm.start_trade_socket(
            callback=self._handle_trade_message, 
            symbol=self.symbol
        )
        print(f"Streams started successfully for {self.symbol}")

    def _handle_depth_message(self, msg):
        """Processes partial book depth updates from the socket."""
        # Ensure message is valid and contains bid/ask updates
        if msg.get('e') == 'depthUpdate' or 'b' in msg:
            # Update local order book cache with current price-quantity pairs
            for b in msg.get('b', []):
                self.order_book['bids'][float(b[0])] = float(b[1])
            for a in msg.get('a', []):
                self.order_book['asks'][float(a[0])] = float(a[1])
            
            # Remove levels where quantity has dropped to 0 to maintain accuracy
            self.order_book['bids'] = {p: v for p, v in self.order_book['bids'].items() if v > 0}
            self.order_book['asks'] = {p: v for p, v in self.order_book['asks'].items() if v > 0}

    def _handle_trade_message(self, msg):
        """Processes individual trade ticks and appends to the rolling buffer."""
        if msg.get('e') == 'trade':
            self.tick_buffer.append({
                'p': float(msg['p']),  # Trade price
                'q': float(msg['q']),  # Trade quantity
                'm': msg['m'],         # Is the buyer the market maker?
                't': msg['T']          # Transaction time in ms
            })

    def get_order_book(self):
        """Returns the sorted top 20 levels of the order book."""
        # Sorted bids (Descending: Highest price first)
        sorted_bids = sorted(self.order_book['bids'].items(), key=lambda x: x[0], reverse=True)[:20]
        # Sorted asks (Ascending: Lowest price first)
        sorted_asks = sorted(self.order_book['asks'].items(), key=lambda x: x[0])[:20]
        
        return {'bids': sorted_bids, 'asks': sorted_asks}

    def stop(self):
        """Stops all active streams and terminates the threaded manager."""
        self.twm.stop()

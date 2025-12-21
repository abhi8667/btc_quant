import time
from binance import ThreadedWebsocketManager
from collections import deque
import numpy as np

class DataStream:
    def __init__(self, client, symbol):
        """
        Initializes the data stream using a pre-configured Binance Client.
        The client object provides API credentials and Testnet settings.
        """
        # All symbols for WebSocket streams MUST be lowercase.
        self.symbol = symbol.lower() 
        self.client = client
        
        # Initialize the manager with the client's credentials.
        # The testnet parameter ensures the manager uses the correct WebSocket URLs.
        self.twm = ThreadedWebsocketManager(
            api_key=client.API_KEY, 
            api_secret=client.API_SECRET,
            testnet=True
            
        )
        
        # Local cache for Order Book (LOB) and trades.
        self.order_book = {'bids': {}, 'asks': {}} 
        self.tick_buffer = deque(maxlen=1000)
        self.last_update_id = 0
        self.data_ready = False
        self.websocket_ready = False

    def _fetch_initial_snapshot(self):
        """Fetches initial order book snapshot from REST API for synchronization."""
        try:
            print(f"Fetching initial order book snapshot for {self.symbol}...")
            depth = self.client.get_order_book(symbol=self.symbol.upper(), limit=20)
            
            # Populate order book with initial snapshot
            self.order_book['bids'] = {float(b[0]): float(b[1]) for b in depth['bids']}
            self.order_book['asks'] = {float(a[0]): float(a[1]) for a in depth['asks']}
            self.last_update_id = depth.get('lastUpdateId', 0)
            
            print(f"Initial snapshot loaded: {len(self.order_book['bids'])} bids, {len(self.order_book['asks'])} asks")
            return True
        except Exception as e:
            print(f"Error fetching initial snapshot: {e}")
            return False

    def start(self):
        """Starts the internal loop and initiates socket connections."""
        # Step 1: Fetch initial snapshot from REST API
        if not self._fetch_initial_snapshot():
            raise RuntimeError("Failed to fetch initial order book snapshot")
        
        # Step 2: CRITICAL: twm.start() must be called BEFORE starting any sockets.
        print("Starting WebSocket manager...")
        self.twm.start() 
        
        # Give the manager a moment to initialize
        time.sleep(1)
        
        # Step 3: Start Partial Book Depth Stream (Top 20 levels).
        # This returns incremental updates that we'll sync with our snapshot.
        print(f"Starting depth stream for {self.symbol}...")
        self.twm.start_depth_socket(
            callback=self._handle_depth_message, 
            symbol=self.symbol,
            depth=20 
        )
        
        # Step 4: Start Trade Stream for micro-tick transaction data.
        print(f"Starting trade stream for {self.symbol}...")
        self.twm.start_trade_socket(
            callback=self._handle_trade_message, 
            symbol=self.symbol
        )
        
        # Step 5: Wait for websocket to receive first update to confirm connection
        print("Waiting for WebSocket synchronization...")
        self._wait_for_websocket_ready()
        
        print(f"Streams successfully synchronized for {self.symbol}")
    
    def _wait_for_websocket_ready(self, timeout=30):
        """Waits for websocket to receive first update, confirming connection."""
        start_time = time.time()
        initial_update_id = self.last_update_id
        
        while not self.websocket_ready:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"WebSocket failed to receive data within {timeout} seconds")
            
            # Check if we've received a websocket update (last_update_id should change)
            # or if we have valid data from the initial snapshot
            if self.last_update_id > initial_update_id or (self.last_update_id > 0 and len(self.order_book['bids']) > 0):
                # Verify we have valid price data
                bids = sorted(self.order_book['bids'].items(), key=lambda x: x[0], reverse=True)
                asks = sorted(self.order_book['asks'].items(), key=lambda x: x[0])
                
                if bids and asks and float(bids[0][0]) > 0 and float(asks[0][0]) > 0:
                    self.websocket_ready = True
                    self.data_ready = True
                    mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
                    print(f"WebSocket synchronized. Current price: {mid_price}")
                    break
            
            time.sleep(0.5)

    def _handle_depth_message(self, msg):
        """Processes depth updates to populate the local cache."""
        try:
            # Check for bid ('b') and ask ('a') updates in the message.
            if 'b' in msg and 'a' in msg:
                # Partial book depth streams (depth=20) return snapshots of top 20 levels
                # Replace the order book with the fresh snapshot for accuracy
                self.order_book['bids'] = {float(b[0]): float(b[1]) for b in msg['b']}
                self.order_book['asks'] = {float(a[0]): float(a[1]) for a in msg['a']}
                
                # Update last update ID for synchronization tracking
                update_id = msg.get('lastUpdateId', msg.get('u', 0))
                if update_id > 0:
                    self.last_update_id = update_id
                    self.websocket_ready = True
                    self.data_ready = True
        except Exception as e:
            print(f"Error handling depth message: {e}")
            pass

    def _handle_trade_message(self, msg):
        """Processes individual trade ticks into the buffer."""
        # Trade events are explicitly marked with event type 'trade'.
        if isinstance(msg, dict) and msg.get('e') == 'trade':
            self.tick_buffer.append({
                'p': float(msg['p']),  # Trade price.
                'q': float(msg['q']),  # Trade quantity.
                'm': msg['m'],         # Is buyer the market maker?.
                't': msg['T']          # Transaction time.
            })

    def get_order_book(self):
        """Returns the sorted top 20 levels required by the environment."""
        # Sorted bids (Descending) and asks (Ascending).
        sorted_bids = sorted(self.order_book['bids'].items(), key=lambda x: x[0], reverse=True)[:20]
        sorted_asks = sorted(self.order_book['asks'].items(), key=lambda x: x[0])[:20]
        return {'bids': sorted_bids, 'asks': sorted_asks}
    
    def is_ready(self):
        """Checks if data stream is ready with valid order book data."""
        return self.data_ready and len(self.order_book['bids']) > 0 and len(self.order_book['asks']) > 0

    def stop(self):
        """Safely terminates all active streams and the manager."""
        # Close all background threads and connection sockets cleanly.
        self.twm.stop()

import time
import random
from settings import LLM_REQUEST_INTERVAL

class RateLimiter:
    """
    A simple rate limiter that ensures a minimum interval between requests.
    Includes a small random jitter to prevent 'thundering herd' issues in 
    multiprocessing environments.
    """
    def __init__(self, interval=None):
        self.interval = interval if interval is not None else LLM_REQUEST_INTERVAL
        self.last_call = 0.0

    def wait(self):
        if self.interval <= 0:
            return

        elapsed = time.time() - self.last_call
        wait_time = max(0, self.interval - elapsed)
        
        # Add a small random jitter (up to 20% of the interval)
        jitter = random.uniform(0, self.interval * 0.2)
        
        if wait_time + jitter > 0:
            time.sleep(wait_time + jitter)
            
        self.last_call = time.time()

# Global rate limiter instance
limiter = RateLimiter()

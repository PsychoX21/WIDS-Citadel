
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os
import math
import heapq

# Add the simulator directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulator'))

from order_book import OrderBook
from engine import MarketEngine
from environment import MarketEnvironment
from logger import Logger
from market_config import MarketConfig
from fair_value import FairValueProcess
from agents import NoiseTraderAgent, MarketMakerAgent, MomentumAgent
from events import AgentArrivalEvent, OrderSubmissionEvent, FairValueUpdateEvent, SnapshotEvent
from actions import PlaceLimit, PlaceMarket, Cancel
from order import Order

class TradingEnv(gym.Env):
    """
    Gymnasium Environment for Citadel Week 3.
    Wraps the existing MarketEngine and simulates background noise traders to create liquidity.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 num_noise_traders=50, 
                 num_market_makers=10, 
                 num_momentum_traders=5, 
                 simulation_time_limit=1800.0, # 30 mins
                 step_duration=1.0, # 1 second per step
                 risk_lambda=0.5, # Risk aversion parameter
                 render_mode=None):
        
        super(TradingEnv, self).__init__()
        
        self.num_noise = num_noise_traders
        self.num_mm = num_market_makers
        self.num_mom = num_momentum_traders
        self.sim_time_limit = simulation_time_limit
        self.step_duration = step_duration
        self.risk_lambda = risk_lambda
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell (Fixed quantity 1 for now, Market orders for simplicity or simple limits)
        # Week 3 docs: "0 -> Hold, 1 -> Buy (fixed size), 2 -> Sell (fixed size)"
        self.action_space = spaces.Discrete(3)

        # Observation Space: Normalized [bids(5), asks(5), bid_vols(5), ask_vols(5), inventory, cash, mid]
        # Top 5 levels for bids and asks prices and volumes = 20 features
        # + Inventory, Cash, Mid = 23 features
        # Actually docs suggest: mid, spread, inventory, cash etc.
        # Let's use a dense representation:
        # 5 Bid Prices relative to Mid
        # 5 Ask Prices relative to Mid
        # 5 Bid Vols (log normalized)
        # 5 Ask Vols (log normalized)
        # Inventory (normalized)
        # Cash (normalized? or just PnL?)
        
        # For simplicity and compliance with "Normalized observation data... between 0 and 1"
        # We will define a fixed size box.
        self.obs_dim = 23 # 5+5+5+5 + 3 (Inv, Cash, Mid?? Mid is absolute, better use returns or relative)
        # Replacing Absolute Mid with Spread or Volatility?
        # Let's stick to standard L2 snapshot.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.engine = None
        
        # RL Agent State
        self.agent_id = "RL_AGENT"
        self.rl_inventory = 0
        self.rl_cash = 100_000.0
        self.initial_cash = 100_000.0
        
        # Reward Tracking
        self.portfolio_history = []
        self.peak_portfolio_value = 100_000.0
        self.prev_portfolio_value = 100_000.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Seed random generators
        self.np_random = np.random.default_rng(seed)
        import random
        random.seed(seed if seed is not None else 42)
        
        # Initialize Simulator Components
        self.book = OrderBook()
        self.logger = Logger() # We might just ignore logging for training speed, or log to /dev/null
        self.engine = MarketEngine(self.book, self.logger)
        self.market_config = MarketConfig(tick_size=1.0)
        self.env_wrapper = MarketEnvironment(self.engine, self.market_config)
        
        self.fv = FairValueProcess(initial_value=100.0, sigma=0.5, seed=seed)
        
        # Setup Background Agents
        self.agents = []
        for i in range(self.num_noise):
            self.agents.append(NoiseTraderAgent(f"N{i}", self.fv, arrival_rate=1.2))
        
        for i in range(self.num_mm):
            self.agents.append(MarketMakerAgent(f"MM{i}", self.fv, arrival_rate=0.2, base_spread=1.0, inventory_skew=0.2))

        for i in range(self.num_mom):
             self.agents.append(MomentumAgent(f"MOM{i}", window=20, arrival_rate=0.5, max_qty=3, cash=100_000))
            
        # Schedule Initial Events
        for agent in self.agents:
            self.engine.agents[agent.agent_id] = agent
            self.engine.schedule(AgentArrivalEvent(agent.next_event_time(0), agent, self.env_wrapper))

        # Schedule Fair Value Updates
        # We need a recurring event for FV update. In run_simulation it was one event?
        # No, FairValueUpdateEvent logic probably schedules the next one? 
        # Checking events.py would confirm, but usually yes. 
        # Actually in run_simulation: engine.schedule(FairValueUpdateEvent(0, fv, dt=1.0))
        # We'll just manually step FV in our loop or schedule it.
        # Let's schedule it to be safe.
        self.engine.schedule(FairValueUpdateEvent(0, self.fv, dt=1.0))
        self.engine.schedule(SnapshotEvent(0, self.env_wrapper, depth=10))

        # Reset RL Agent
        self.rl_inventory = 0
        self.rl_cash = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        self.portfolio_history = [self.initial_cash]
        
        # Warmup: Run engine for 60 seconds to populate book
        self._run_engine_until(60.0)
        
        return self._get_obs(), {}

    def _run_engine_until(self, target_time):
        """
        Run the engine until the simulation time reaches target_time.
        """
        while self.engine.event_queue:
            # peek next event
            evt_time, _, _ = self.engine.event_queue[0]
            
            if evt_time > target_time:
                break
                
            # Pop and execute
            _, _, event = heapq.heappop(self.engine.event_queue)
            self.engine.time = evt_time
            event.execute(self.engine)
            
        self.engine.time = target_time

    def step(self, action):
        # 1. Execute RL Action
        # Action 0: Hold
        # Action 1: Buy (Market 1 unit)
        # Action 2: Sell (Market 1 unit)
        
        current_mid = self.env_wrapper.get_market_state()["mid"]
        if current_mid is None:
            current_mid = self.fv.get() # Fallback

        # Execute Trade
        # Directly interacting with matching engine logic effectively
        # Or simpler: Just PlaceMarket via order book?
        # We need to act as an agent.
        
        trade_executed = False
        trade_price = 0
        
        # Create an Order object for RL agent
        # We will submit it immediately.
        
        rl_order = None
        if action == 1: # BUY
            rl_order = Order(order_id=f"RL-{self.engine.time}", side="BUY", price=None, qty=1, timestamp=self.engine.time)
        elif action == 2: # SELL
            rl_order = Order(order_id=f"RL-{self.engine.time}", side="SELL", price=None, qty=1, timestamp=self.engine.time)
            
        if rl_order:
            # We match immediately against current book
            # Check liquidity
            snapshot = self.book.current_snapshot()
            
            # Simple manual matching for RL agent to get fill price immediately
            # In a real event system, we'd schedule an event. 
            # But step() implies atomic decision.
            # We will use book.submit(rl_order) logic.
            # However, we need to capture execution price to update RL cash.
            
            # Since book.submit() invokes _match() and stores trades in self.trades...
            # We can check existing trades count, submit, then check new trades.
            
            pre_trades_len = len(self.book.trades)
            self.book.submit(rl_order)
            post_trades_len = len(self.book.trades)
            
            # Calculate fill info
            if post_trades_len > pre_trades_len:
                new_trades = self.book.trades[pre_trades_len:]
                # Filter for our order
                for t in new_trades:
                    if (rl_order.side == "BUY" and t.buy_order_id == rl_order.order_id) or \
                       (rl_order.side == "SELL" and t.sell_order_id == rl_order.order_id):
                        
                        trade_executed = True
                        trade_price = t.price
                        # Update Cash/Inventory
                        if rl_order.side == "BUY":
                            self.rl_inventory += t.qty
                            self.rl_cash -= t.price * t.qty
                        else:
                            self.rl_inventory -= t.qty
                            self.rl_cash += t.price * t.qty
                            
        # 2. Advance Time (Background Market)
        next_time = self.engine.time + self.step_duration
        self._run_engine_until(next_time)

        # 3. Calculate Reward
        # Day 3 Formula: Reward = (V_t - V_{t-1}) - lambda * max(0, Peak_t - V_t)
        
        # Mark to Market Portfolio Value
        # Inventory valued at Mid Price
        snap = self.book.current_snapshot()
        best_bid = snap.best_bid()
        best_ask = snap.best_ask()
        mid_price = None
        if best_bid is not None and best_ask is not None:
             mid_price = (best_bid + best_ask) / 2.0
        
        if mid_price is None:
            mid_price = self.fv.get() # Fallback
            
        current_portfolio_value = self.rl_cash + (self.rl_inventory * mid_price)
        
        # Update Peak
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_portfolio_value)
        
        # Components
        pnl = current_portfolio_value - self.prev_portfolio_value
        drawdown_penalty = self.risk_lambda * max(0, self.peak_portfolio_value - current_portfolio_value)
        
        reward = pnl - drawdown_penalty
        
        self.prev_portfolio_value = current_portfolio_value
        self.portfolio_history.append(current_portfolio_value)
        
        # 4. Check Termination
        terminated = False
        truncated = False
        
        if self.engine.time >= self.sim_time_limit:
            terminated = True
            
        # Bankrupt check
        if current_portfolio_value <= 0:
            terminated = True
            reward -= 1000 # Heavy penalty for ruin

        # 5. Observation
        obs = self._get_obs(mid_price)
        
        info = {
            "portfolio_value": current_portfolio_value,
            "inventory": self.rl_inventory,
            "pnl_step": pnl,
            "drawdown": self.peak_portfolio_value - current_portfolio_value
        }
        
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self, mid_price=None):
        snapshot = self.book.current_snapshot()
        
        if mid_price is None:
            best_bid = snapshot.best_bid()
            best_ask = snapshot.best_ask()
            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2.0
            else:
                mid_price = 100.0
            
        # Feature Engineering (Vectorized)
        # Bids: Top 5 prices (relative to mid) and volumes (log)
        # Asks: Top 5 prices (relative to mid) and volumes (log)
        
        def safe_log(x):
            return np.log(x + 1)
            
        # snapshot.bids is list of (price, qty)
        bids = snapshot.bids[:5] 
        asks = snapshot.asks[:5] 
        
        # Pad if less than 5
        bid_feats = np.zeros(10) # 5 prices, 5 vols
        ask_feats = np.zeros(10)
        
        for i, b in enumerate(bids):
            p = b[0]
            v = b[1]
            bid_feats[i] = (p - mid_price) / mid_price # Relative Price
            bid_feats[i+5] = safe_log(v) # Log Vol
            
        for i, a in enumerate(asks):
            p = a[0]
            v = a[1]
            ask_feats[i] = (p - mid_price) / mid_price
            ask_feats[i+5] = safe_log(v)
            
        # Global State
        # Inventory (normalized by max_inventory assumption, say 100)
        inv_feat = self.rl_inventory / 100.0 
        
        # Cash? Maybe insignificant compared to Invested? 
        # Usually Inventory + Recent Returns
        # Let's use Cash / Initial
        cash_feat = self.rl_cash / self.initial_cash
        
        # Create vector
        obs = np.concatenate([bid_feats, ask_feats, [inv_feat, cash_feat, 0.0]]) # 23 dim, last one unused/padding
        
        # Normalize/Clip to reasonable range 
        # Since we use Box(-inf, inf), we don't strictly need to clip, 
        # provides raw relative features.
        
        return obs.astype(np.float32)

    def render(self, mode='human'):
        print(f"Time: {self.engine.time:.2f} | PF: {self.prev_portfolio_value:.2f} | Inv: {self.rl_inventory}")

# Self-Check
if __name__ == "__main__":
    env = TradingEnv()
    obs, _ = env.reset()
    print("Initial Obs:", obs.shape)
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        print(f"Rew: {r:.4f} | Info: {info}")
        if term: break

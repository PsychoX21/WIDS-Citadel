
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure simulator is in path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulator'))

from stable_baselines3 import PPO
from trading_env import TradingEnv

def plot_stylized_facts(l1_df, output_dir):
    """
    Generate plots for Volatility Clustering and Fat Tails.
    """
    if l1_df.empty:
        print("No L1 data to plot.")
        return

    # 1. Prepare Returns
    mid_prices = l1_df.set_index("time")["mid"]
    # Resample to 1-second intervals to get consistent returns
    # But simulation time might not be perfectly grid aligned.
    # Forward fill is safer.
    # Create a grid
    grid = np.arange(0, mid_prices.index.max(), 1.0)
    mid_resampled = mid_prices.reindex(mid_prices.index.union(grid)).ffill().reindex(grid)
    
    log_returns = np.log(mid_resampled).diff().dropna()
    
    # 2. Fat Tails (Histogram vs Normal)
    plt.figure(figsize=(10, 6))
    sns.histplot(log_returns, kde=True, stat="density", label="Returns")
    
    # Plot Normal Distribution with same mean/std
    mu, std = log_returns.mean(), log_returns.std()
    x = np.linspace(min(log_returns), max(log_returns), 100)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
    plt.plot(x, p, 'r--', linewidth=2, label="Normal Dist")
    
    plt.title("Return Distribution (Fat Tails Check)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "fat_tails.png"))
    plt.close()
    
    # 3. Volatility Clustering (ACF of Abs Returns)
    # Filter out zero returns which are artifacts of sampling frequency
    clean_returns = log_returns[log_returns != 0]
    abs_returns = np.abs(clean_returns)
    from statsmodels.tsa.stattools import acf
    
    # Calculate ACF manually if statsmodels not present, but usually implied by data science stack
    # Let's try/except
    try:
        acf_vals = acf(abs_returns, nlags=20)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(acf_vals)), acf_vals)
        plt.title("Autocorrelation of Absolute Returns (Volatility Clustering)")
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.savefig(os.path.join(output_dir, "vol_clustering.png"))
        plt.close()
    except ImportError:
        print("statsmodels not found, skipping ACF plot.")

def plot_heatmap(l2_data, output_dir):
    """
    Generate Limit Order Book Heatmap.
    """
    if not l2_data:
        print("No L2 data for heatmap.")
        return
        
    # Data is list of {time, bids, asks}
    # We want a DataFrame: Index=Price, Columns=Time, Values=Volume
    
    # 1. Collect all prices and times
    # This can be huge. We need to bin.
    
    times = [d['time'] for d in l2_data]
    
    # Flatten all orders to find price range
    all_bids = []
    all_asks = []
    
    # Optimize: Process step by step
    # We'll create a sparse matrix? Or just scatter plot for "Liquidity Walls"?
    # Heatmap is better.
    
    # Let's target a price grid.
    # Current mid is ~100.
    
    # Collect data for dataframe construction
    data_list = []
    for snapshot in l2_data:
        t = snapshot['time']
        for p, qty in snapshot['bids']:
            data_list.append({'time': t, 'price': p, 'qty': qty, 'side': 'bid'})
            
        for p, qty in snapshot['asks']:
            data_list.append({'time': t, 'price': p, 'qty': qty, 'side': 'ask'})
            
    df = pd.DataFrame(data_list)
    if df.empty:
        print("DF Empty for Heatmap")
        return

    # Pivot? 
    # Bin prices to integer ticks
    df['price_bin'] = df['price'].astype(int)
    # Bin times to 10s
    df['time_bin'] = (df['time'] // 10) * 10
    
    # Sum volume
    pivot = df.groupby(['price_bin', 'time_bin'])['qty'].sum().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="viridis", robust=True)
    plt.title("Limit Order Book Liquidity Heatmap")
    plt.xlabel("Time (10s bins)")
    plt.ylabel("Price")
    plt.savefig(os.path.join(output_dir, "lob_heatmap.png"))
    plt.close()

def main():
    print("Initializing Multi-Agent Simulation (Day 6/7)...")
    
    # Create Output Dir
    output_dir = "Week 3/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Init Env
    env = TradingEnv(
        num_noise_traders=50,
        num_market_makers=10,
        num_momentum_traders=10, # Added Momentum Traders
        simulation_time_limit=5100.0, 
        step_duration=1.0,
        render_mode=None
    )
    
    # Load Agent
    model_path = os.path.join("Week 3", "ppo_trading_agent_v1")
    model = None
    if os.path.exists(model_path + ".zip"):
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path)
    else:
        print(f"Model not found at {model_path}. Using Random Agent.")
    
    print("Running Simulation...")
    obs, _ = env.reset()
    done = False
    step_cnt = 0
    
    while not done:
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_cnt += 1
        
        if step_cnt % 100 == 0:
            print(f"Step {step_cnt}/5000", end="\r")
            
    print(f"\nSimulation Complete. Steps: {step_cnt}")
    
    # Extract Data
    print("Extracting Data...")
    l1_df = env.logger.l1_df()
    l2_data = env.logger.l2
    trades_df = env.logger.trades_df()
    
    # Save Data
    l1_df.to_csv(os.path.join(output_dir, "l1_data.csv"), index=False)
    trades_df.to_csv(os.path.join(output_dir, "trades.csv"), index=False)
    
    # Plotting
    print("Generating Plots...")
    plot_stylized_facts(l1_df, output_dir)
    plot_heatmap(l2_data, output_dir)
    
    print("Week 3 Complete. Artifacts in Week 3/results/")

if __name__ == "__main__":
    main()

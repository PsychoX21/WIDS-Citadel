
import gymnasium as gym
import os
import sys

# Ensure simulator is in path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulator'))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from trading_env import TradingEnv
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Checking Environment...")
    # Reduce complexity for quick verification
    env = TradingEnv()
    check_env(env)
    print("Environment Verified.")
    
    print("Starting PPO Training (Verification - Short Run)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64
    )
    
    # Train for only 2048 timesteps to prove it works and saves
    total_timesteps = 2048
    model.learn(total_timesteps=total_timesteps)
    
    print("Training Complete.")
    save_path = os.path.join("Week 3", "ppo_trading_agent_v1")
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()


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
    env = TradingEnv()
    check_env(env)
    print("Environment Verified.")
    
    print("Starting PPO Training (Day 5 Sanity Check)...")
    # PPO Parameters per Week 3 instructions
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        batch_size=64,
        tensorboard_log="./ppo_tensorboard/"
    )
    
    # Train for 50,000 timesteps
    total_timesteps = 50000
    model.learn(total_timesteps=total_timesteps)
    
    print("Training Complete.")
    model.save("ppo_trading_agent_v1")
    
    # Verify Performance (Run one episode)
    print("\nRunning Verification Episode...")
    obs, _ = env.reset()
    done = False
    portfolio_values = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info['portfolio_value'])
        done = terminated or truncated
        
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values)
    plt.title("RL Agent Portfolio Value (Validation Run)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.savefig("validation_run.png")
    print("Validation plot saved to validation_run.png")

if __name__ == "__main__":
    main()

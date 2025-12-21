# RL Crypto Trading Bot

A Reinforcement Learning (RL) crypto trading bot for Binance Spot market using PPO algorithm, designed to trade micro-changes.

## Features

- Custom Gymnasium environment for Binance order book data
- PPO agent using Stable Baselines3
- Real-time WebSocket data streaming from Binance
- Modular architecture for training and live trading
- Micro-trading focused on small position changes


Phase 1: Market "Code Cracking" (High Priority)
[1] Improve Explained Variance: Current variance is negative (-2.58), meaning the model doesn't yet understand price causality.Action: Introduce Bollinger Bands or MACD to the observation space.
[2] Adjust Entropy Coefficient: Current exploration is dropping too early.Action: Increase ent_coef from $0.01$ to $0.03$ to force the AI to search for winning strategies longer.
Phase 2: Strategy Fine-Tuning
[1] Reward Shaping: Increase the Realized Profit Multiplier.Action: Shift from $50\times$ to $100\times$ profit multiplier to make "winning" signals stand out more against fees.
[2] Implement Model Checkpoints: Ensure the bot saves every 50k steps to prevent data loss during long runs.
Phase 3: Real-Time Testing
[1] Evaluation Script: Create a standalone script to run the saved .zip model on 10,000 steps of fresh test data without learning.
[2] Drawdown Protection: Implement a hard-stop logic if the balance drops more than 15% in a single session.

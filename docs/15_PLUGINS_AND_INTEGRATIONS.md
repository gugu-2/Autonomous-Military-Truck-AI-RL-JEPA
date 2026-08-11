# Third-Party Plugins & Integrations

The OMNIDRIVE stack stands on the shoulders of giants. Rather than re-inventing the wheel, we deeply integrated five major state-of-the-art open-source plugins to handle specific domains of the architecture. This document provides a detailed analysis of how each plugin is utilized.

## 1. Meta V-JEPA & Drive-JEPA (Perception)
* **What it is:** Meta's self-supervised video feature extractor.
* **How it is used:** It forms the backbone of Layer 2 (Perception). We modified the `vit_encoder.py` to accept driving-specific frame rates and multimodal (LiDAR+Radar) cross-attention, but the core weights are initialized using Meta's pre-trained V-JEPA models.
* **Why we chose it:** It completely eliminates the need for manual data labeling (e.g., drawing bounding boxes around pedestrians). It learns object permanence and physics simply by predicting missing pixels in video streams.

## 2. NVIDIA Alpamayo OpenMDW-1.1 (Reasoning)
* **What it is:** A massively capable Vision-Language-Action (VLA) foundation model designed for robotics.
* **How it is used:** In Layer 4, the Alpamayo client runs locally via a PyTorch API. When the RL controller encounters a scenario it does not understand (e.g., a flooded road or a military checkpoint), the camera frame is passed to Alpamayo.
* **Why we chose it:** Traditional RL policies cannot generalize to situations they haven't seen millions of times. Alpamayo acts as the "Zero-Shot" reasoning engine, capable of understanding human gestures, written signs, and complex physical obstructions without prior driving-specific training.

## 3. DreamerV3 & CarDreamer (Action Policy)
* **What it is:** DeepMind's flagship Model-Based Reinforcement Learning algorithm.
* **How it is used:** Located in Layer 5. The core `RSSM` (Recurrent State Space Model) tracks the driving environment's latent state. The Actor-Critic networks are trained to navigate by maximizing a complex reward function.
* **Why we chose it:** Model-free RL (like PPO or SAC) requires millions of physical driving miles to learn not to crash. DreamerV3 learns to drive by hallucinating trajectories entirely inside the JEPA World Model, meaning it can learn to drive in hours without ever touching a real car.

## 4. Autoware.Universe (Navigation & Rule Engine)
* **What it is:** The world's leading open-source autonomous driving framework built on ROS 2.
* **How it is used:** Layer 6. We ripped out Autoware's perception and control modules (replacing them with JEPA and DreamerV3, respectively). However, we kept Autoware's HD Map Loader (Lanelet2), NDT Localizer, and Global Route Planner.
* **Why we chose it:** Neural networks are terrible at strict localization and following static maps. Autoware provides mathematical precision for intersection management, GPS routing, and adherence to static traffic laws.

## 5. CARLA (Simulation & Testing)
* **What it is:** An open-source simulator for autonomous driving research, utilizing Unreal Engine.
* **How it is used:** During the training phase, the `carla_env_wrapper.py` interfaces with the CARLA Python API. The RL agent receives synthetic dashcam video and LiDAR from CARLA, and sends back steering/throttle commands.
* **Why we chose it:** CARLA allows us to simulate blizzards, military off-road environments, aggressive pedestrian traffic, and sensor failure modes in a perfectly safe virtual environment before deploying to physical hardware.

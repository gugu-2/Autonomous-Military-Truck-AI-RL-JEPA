# Current Limitations & Unsolved Problems

To maintain project scope, ensure absolute safety, and respect physical hardware limitations, we have explicitly chosen **not** to solve the following problems. This document details the boundary limitations of the OMNIDRIVE system.

## 1. High-Speed Racing & Evasive Physics (>120 km/h)
**The Limitation:** OMNIDRIVE cannot safely drive a vehicle at extremely high speeds or perform professional evasive driving maneuvers (like a J-turn or high-speed drifting).
**Detailed Analysis:** 
* The RL controller's reward functions and the JEPA 3-second imagination horizon are optimized for military convoys, logistics trucks, and urban robotaxis. 
* At 150 km/h, 3 seconds of imagination only covers 125 meters of distance. For heavy vehicles, this is not enough stopping distance.
* Expanding the imagination horizon to 6 seconds ($k=20$) requires exponentially more memory, which immediately causes a VRAM Out-of-Memory (OOM) crash on the 8GB RTX 4050. Therefore, the system is permanently speed-capped.

## 2. Non-Deterministic GPU Safety Guarantees
**The Limitation:** We cannot mathematically guarantee that the deep learning models will never hallucinate a ghost object or fail to see a real one.
**Detailed Analysis:**
* Neural networks operate on statistical probabilities, not absolute boolean logic. The "black box" nature of ViT encoders means a specifically crafted adversarial physical object could theoretically blind the perception engine.
* **Why we left this unsolved in AI:** We intentionally left this "unsolved" at the GPU level because we solved it at the CPU level. We rely entirely on the `SafetyMonitor` (Layer 7) to act as a hardcoded physics overseer. If the AI hallucinates and requests full throttle into a wall, the CPU physics check overrides the GPU and brakes.

## 3. Artificial General Intelligence (AGI) Freedom
**The Limitation:** The reasoning module (Alpamayo) is incredibly smart, but it is heavily constrained in what it is allowed to do. It cannot freely talk to the driver or directly control the steering wheel.
**Detailed Analysis:**
* Connecting an LLM directly to the steering rack introduces latency spikes (LLMs take hundreds of milliseconds to generate tokens) and catastrophic failure risk if the LLM hallucinates an invalid command.
* Therefore, the LLM is restricted to a strict intent-parsing dictionary (`STOP`, `YIELD`, `REROUTE`). It operates in a read-only asynchronous loop, meaning it can only "suggest" hints to the RL controller, rather than actively driving the car.

## 4. Hardware-Agnostic Plug-and-Play
**The Limitation:** You cannot take this software and plug it into any random car without severe calibration.
**Detailed Analysis:**
* The RL agent's latent space is trained on a specific vehicle's kinematics (mass, wheelbase, turn radius, brake force).
* A policy trained on a 3,000lb Tesla will catastrophically fail if deployed on a 15,000lb Military transport truck because the brake timing will be completely wrong. Retraining in CARLA using the specific vehicle's physics profile is strictly required.

# OMNIDRIVE Layer 4: Reasoning Module Technical Specification
## NVIDIA Alpamayo Vision-Language-Action (VLA) System Integration

---

### Document Metadata
- **System Layer:** Layer 4 — Cognitive Reasoning & Long-Tail VLA System
- **Module Identifier:** `OMNIDRIVE-REASONING-VLA`
- **Target File Path:** `OMNIDRIVE_PROJECT/docs/04_REASONING_MODULE.md`
- **Primary Dependencies:** Layer 1 (Perception & Sensors), Layer 2 (JEPA World Model), Layer 3 (RL Motion Controller), Layer 5 (Safety Shield & Guardrails)
- **Execution Model:** Asynchronous Thread Execution (10 Hz / ~100ms latency budget), decoupled from Layer 3 Real-Time Control Loop (83.3 Hz / 12ms tick rate)
- **License:** OpenMDW-1.1 (Open Mobility Driving License)

---

## Table of Contents
1. [Module Overview](#1-module-overview)
2. [The Long-Tail Problem in Autonomous Driving](#2-the-long-tail-problem-in-autonomous-driving)
3. [NVIDIA Alpamayo Model Specifications](#3-nvidia-alpamayo-model-specifications)
4. [Alpamayo Client Design (`alpamayo_client.py`)](#4-alpamayo-client-design-alpamayo_clientpy)
5. [Scene Descriptor Generator (`scene_descriptor.py`)](#5-scene-descriptor-generator-scene_descriptorpy)
6. [VLA Reasoner Engine (`vla_reasoner.py`)](#6-vla-reasoner-engine-vla_reasonerpy)
7. [Rare Scenario Handler (`rare_scenario_handler.py`)](#7-rare-scenario-handler-rare_scenario_handlerpy)
8. [Scenario Classifier (`scenario_classifier.py`)](#8-scenario-classifier-scenario_classifierpy)
9. [Intent Parser (`intent_parser.py`)](#9-intent-parser-intent_parserpy)
10. [Command Translator (`command_translator.py`)](#10-command-translator-command_translatorpy)
11. [Military Scenario Handling Specifications](#11-military-scenario-handling-specifications)
12. [Heavy Truck & Commercial Vehicle Scenario Specifications](#12-heavy-truck--commercial-vehicle-scenario-specifications)
13. [Asynchronous Multi-Threaded Execution Architecture](#13-asynchronous-multi-threaded-execution-architecture)
14. [Integration Architecture & System Dataflow](#14-integration-architecture--system-dataflow)
15. [Configuration Schema](#15-configuration-schema)
16. [API Interface & Python Class Stubs](#16-api-interface--python-class-stubs)
17. [Unit Test Plan & Verification Suite](#17-unit-test-plan--verification-suite)
18. [Appendix: Example Input/Output JSON Schemas & Payloads](#18-appendix-example-inputoutput-json-schemas--payloads)

---

## 1. Module Overview

The **Reasoning Module** (Layer 4 of the OMNIDRIVE 7-layer autonomous driving architecture) bridges the critical gap between sub-symbolic spatial motion control and high-level cognitive context understanding. While Layer 2 (JEPA World Model) predicts local spatial dynamics and Layer 3 (RL Motion Controller) generates continuous control commands (steering, throttle, braking) at 83.3 Hz (12ms), Layer 4 provides **zero-shot and few-shot semantic reasoning over out-of-distribution (OOD) and long-tail scenarios**.

```
+-----------------------------------------------------------------------------------+
|                            OMNIDRIVE 7-LAYER ARCHITECTURE                         |
+-----------------------------------------------------------------------------------+
| Layer 1: Sensor Perception & Feature Extraction (Camera, LiDAR, Radar, IMU)       |
| Layer 2: JEPA World Model (Latent Space Dynamics & Spatial Hazard Energy Map)      |
| Layer 3: RL Motion Controller (Continuous Trajectory & Control Loop @ 12ms / 83Hz) |
+-----------------------------------------------------------------------------------+
| Layer 4: REASONING MODULE (NVIDIA Alpamayo VLA System @ 100ms / 10Hz Async)       |  <-- THIS MODULE
+-----------------------------------------------------------------------------------+
| Layer 5: Safety Shield & Hard Constraints (Deterministic Barrier Functions)        |
| Layer 6: Fleet Telemetry & Multi-Agent Tactical Coordination                      |
| Layer 7: Mission Planning & Global Route Optimization                             |
+-----------------------------------------------------------------------------------+
```

### 1.1 What is NVIDIA Alpamayo?
NVIDIA Alpamayo is an advanced Vision-Language-Action (VLA) foundation model designed specifically for autonomous physical agents. Built upon high-capacity multi-modal Vision-Language Architectures, Alpamayo processes multi-view camera frames, spatial hazard maps, and contextual text prompts, outputting structured Chain-of-Thought (CoT) reasoning along with discrete **Action Tokens** and driving intents.

### 1.2 The Need for VLA Reasoning
Pure neural networks (such as End-to-End Imitation Learning or Model-Based RL over raw sensor/latent embeddings) excel at nominal driving maneuvers (lane keeping, standard car following, turn signal response). However, they suffer from **epistemic opacity** when encountering novel visual configurations, non-verbal human gestures, procedural regulations, or emergency signals. 

The Reasoning Module acts as an asynchronous **Cognitive Co-Pilot**:
- Monitors the driving environment continuously via lightweight trigger logic.
- Executes deep multi-modal reasoning when unexpected spatial hazards or low-confidence states are detected.
- Emits high-level **Driving Intents** (e.g., `STOP`, `YIELD`, `REROUTE`, `SLOW_DOWN`) and spatial bias vectors to Layer 3, temporarily re-shaping the RL controller's reward surface and action bounds without disrupting real-time latency guarantees.

---

## 2. The Long-Tail Problem in Autonomous Driving

Autonomous vehicles operating in unconstrained environments encounter the **Long-Tail Distribution**: an infinite spectrum of low-frequency, high-severity edge cases that are statistically unrepresented in offline training datasets.

```
 Event
 Frequency
   ^
   |  +------------------------+
   |  | Nominal Driving        |
   |  | (Lane following,       |  99.9% of driving time (Covered by JEPA + RL)
   |  |  adaptive cruise, etc.)|
   |  +------------------------+
   |                            \
   |                             \  The Long Tail (0.1% of scenarios)
   |                              \ (Covered by Layer 4 Alpamayo VLA)
   |                               +--------------------------------------------+
   |                               | Flooded Roads | Hand Signals | Checkpoints |
   +-------------------------------+--------------------------------------------+--> Edge Case Complexity
```

### 2.1 Why Pure JEPA + RL Fails on Rare Scenarios

1. **JEPA Energy Landscape Flatness / Epistemic Noise:**
   The JEPA World Model predicts latent state transitions by minimizing prediction error:
   $$\mathcal{E}(s, a) = \| \mathbf{z}_{\text{target}} - s_{\theta}(\mathbf{z}_{\text{context}}, a) \|_2^2$$
   When an out-of-distribution visual artifact appears (e.g., a soldier holding a red flag), the JEPA encoder projects the input into an uncalibrated region of latent space. The resulting energy landscape $\mathcal{E}(s, a)$ becomes noisy or flat, preventing gradient-based trajectory optimization from finding a safe trajectory.

2. **RL Policy Entropy & Overconfidence Collapse:**
   Reinforcement Learning policies $\pi_{\phi}(a|s)$ trained on standard driving logs learn direct mappings from perception features to action outputs. When presented with conflicting signals (e.g., a green traffic light paired with a police officer waving hand signals to stop), the policy either exhibits high entropy (oscillating controls) or overconfident catastrophic failure (ignoring the human controller).

3. **Absence of Symbolic / Procedural Context:**
   Pure neural policies cannot parse symbolic rules like *"Stop 50 meters before the barrier, wait for the green hand torch signal, then proceed at <5 km/h."* Such multi-step procedural logic requires explicit linguistic abstraction and temporal context execution.

### 2.2 Deep Analysis of 7 Target Long-Tail Scenarios

| Scenario ID | Rare Scenario Description | Failure Mode of Pure JEPA + RL | Alpamayo VLA Solution Strategy |
| :--- | :--- | :--- | :--- |
| **LT-01** | **Soldier / Traffic Controller Waving to Stop** | Person classified as static pedestrian; hand signal ignored; vehicle proceeds through crosswalk. | Identifies military uniform, arm angle vector, interprets hand posture as mandatory stop command, injects zero-velocity intent. |
| **LT-02** | **Flooded Road / Submerged Asphalt** | Depth camera/LiDAR reflections create false planar ground; vehicle drives into deep water, damaging power electronics. | Parses visual surface reflection + water ripple text context, measures depth ambiguity, triggers `REROUTE` intent. |
| **LT-03** | **Roadblock / Unofficial Debris Barrier** | Bounding box detector fails to classify irregular objects (downed trees, concrete blocks, spikes). | Scene descriptor identifies obstruction cluster; VLA reasons about physical non-passability, issues `HALT_AND_PLAN_DETOUR`. |
| **LT-04** | **Construction Zone with Manual Flagging** | Static map contradicts temporary cone layout and flagger paddle; RL agent freezes or hits cones. | Overrides static HD map with visual flagger paddle state ("STOP" / "SLOW"); applies dynamic corridor bias to RL policy. |
| **LT-05** | **Emergency Vehicle Counter-Flow Approach** | Vehicle fails to yield right-of-way when emergency vehicle approaches from opposite lane or behind against red light. | Detects flashing siren light pattern and directional arrows; issues immediate `PULL_TO_MARGIN_AND_YIELD` intent. |
| **LT-06** | **Animal Crossings / Livestock Herds** | Non-standard entity morphology causes bounding box jitter and erratic trajectory predictions in Layer 2. | Recognizes animal herd behavior; projects conservative standoff corridor; holds vehicle stationary until herd clears. |
| **LT-07** | **Military Checkpoints & Armed Guard Posts** | Vehicle fails to comply with multi-stage stopping distances, guard gestures, or speed limits (<5 km/h). | Executes 4-stage checkpoint state machine guided by VLA visual verification at each approach boundary. |

---

## 3. NVIDIA Alpamayo Model Specifications

The OMNIDRIVE Reasoning Module integrates two deployment tiers of the NVIDIA Alpamayo model family, allowing dynamic scalability based on vehicle hardware configurations (e.g., dual NVIDIA DRIVE Thor vs single AGX Orin).

```
+-----------------------------------------------------------------------------------+
|                         NVIDIA ALPAMAYO MODEL FAMILY SPECS                        |
+------------------------------------+----------------------------------------------+
| Specification Parameter            | Value / Implementation Details               |
+------------------------------------+----------------------------------------------+
| Model Name Variants                | Alpamayo 1 (Base) / Alpamayo 2 Super         |
| HuggingFace Hub Source             | nvidia/alpamayo-1-8b-vla                     |
|                                    | nvidia/alpamayo-2-70b-super                  |
| License                            | OpenMDW-1.1 (Open Mobility Driving License)  |
| Vision Encoder Backbone            | Multi-Camera Spatial Transformer (ViT-H/14)  |
| Language/Reasoning Core            | Decoder-Only Autoregressive VLM              |
| Action Token Head                  | 256 / 512 Vectorized Spatial Action Tokens    |
| Native Quantization Support        | FP8 (E4M3 / E5M2), INT8 (W8A8), FP16         |
| Context Window Size                | 8,192 Tokens                                 |
+------------------------------------+----------------------------------------------+
```

### 3.1 Model Tier Breakdown

```
+-----------------------------------------------------------------------------------+
| ALPAMAYO 1 (8B Base Model)                                                        |
+-----------------------------------------------------------------------------------+
| Total Parameters:   8.4 Billion                                                   |
| VRAM Footprint:     16 GB (FP8 Quantized with TensorRT-LLM)                       |
| Primary Hardware:   NVIDIA AGX Orin / Single DRIVE Thor                           |
| Target Latency:     45ms - 65ms per inference pass                                |
| Action Vocabulary:  256 discrete action tokens + JSON Intent output               |
| Best Used For:      Standard edge-case reasoning, urban hand signals, roadblocks  |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
| ALPAMAYO 2 SUPER (70B Flagship Model)                                             |
+-----------------------------------------------------------------------------------+
| Total Parameters:   70.6 Billion                                                  |
| VRAM Footprint:     72 GB (FP8 Multi-GPU Tensor Parallelism)                      |
| Primary Hardware:   Dual NVIDIA DRIVE Thor / Server-grade H100/A100 compute node  |
| Target Latency:     85ms - 110ms per inference pass                               |
| Action Vocabulary:  512 fine-grained action tokens + multi-agent CoT reasoning    |
| Best Used For:      Complex military operations, tactical convoying, heavy truck  |
|                     multi-hazard navigation, procedural state machine management  |
+-----------------------------------------------------------------------------------+
```

### 3.2 Input & Output Modality Specification

1. **Visual Inputs:**
   - Up to 6x Surround RGB Camera Streams (1920x1080 @ 30 FPS), resized/cropped to $448 \times 448$ patch grids per view.
   - JEPA Spatial Hazard Energy Map rendered as a 2D thermal overlay in camera perspective space.

2. **Linguistic & State Inputs:**
   - Structured Scene Prompt (generated by `scene_descriptor.py`).
   - Current Vehicle Telemetry: Speed vector $\mathbf{v} = (v_x, v_y)$, Yaw rate $\dot{\psi}$, Steering Angle $\delta$, Active Navigation Route Segment.

3. **Outputs:**
   - **Reasoning Chain (Chain-of-Thought):** Natural language explanation of detected hazard and logical deduction.
   - **Action Tokens:** Discrete spatial trajectory control hints ($\mathbf{x}_{\text{target}}, \mathbf{y}_{\text{target}}, v_{\text{target}}$).
   - **Structured Intent Payload:** Standardized JSON object containing `primary_intent`, `standoff_distance`, `target_velocity`, `corridor_offset`, and `confidence`.

---

## 4. Alpamayo Client Design (`alpamayo_client.py`)

The `AlpamayoClient` class serves as the direct software interface to the local Alpamayo runtime engine, managing model loading, TensorRT execution streams, KV-cache memory pools, and inference execution.

```
+-----------------------------------------------------------------------------------+
|                          ALPAMAYO CLIENT INFERENCE PIPELINE                       |
+-----------------------------------------------------------------------------------+
| Raw Cameras + JEPA Map + Telemetry                                                |
|       |                                                                           |
|       v                                                                           |
| [ Image Preprocessor & Tokenizer ] --> Patchify images to CUDA memory (8ms)       |
|       |                                                                           |
|       v                                                                           |
| [ Async TensorRT-LLM Engine Stream ]                                              |
|   +-- Vision Transformer Pass (25ms)                                              |
|   +-- Speculative LLM Decoder Pass (FP8 KV Cache) (55ms)                          |
|       |                                                                           |
|       v                                                                           |
| [ Raw Token & Text Output Stream ]                                                |
|       |                                                                           |
|       v                                                                           |
| [ Regex & JSON Schema Parser ] --> Extracted Structured Intent Payload (5ms)     |
|                                    TOTAL LATENCY: ~93ms (<100ms Budget)           |
+-----------------------------------------------------------------------------------+
```

### 4.1 Local Loading & Optimization Architecture
- **Engine Initialization:** Loads pre-compiled TensorRT engines (`.plan`) built from HuggingFace `nvidia/alpamayo-1-8b-vla` or `nvidia/alpamayo-2-70b-super`.
- **Memory Pinning & CUDA Streams:** Allocates pinned host memory buffers for camera frames and binds inference to a dedicated non-blocking CUDA Stream (`cudaStreamNonBlocking`).
- **Graph Capture:** Captures vision encoder forward execution as static CUDA Graphs to eliminate CPU enqueue overhead.

### 4.2 Latency Budget Breakdown (<100ms Target)

```
+-----------------------------------------------------------------------------------+
| STAGE                                    | BUDGET   | OPTIMIZATION TECHNIQUE      |
+------------------------------------------+----------+-----------------------------+
| 1. Image Downsampling & Tensor Format    | 8 ms     | Hardware NvMedia / CUDA IPC |
| 2. Scene Description Prompt Formatting   | 2 ms     | Zero-copy string builder    |
| 3. Vision Transformer Encoding (ViT)     | 25 ms    | FP8 Tensor Cores + TRT      |
| 4. Autoregressive LLM Action Generation  | 55 ms    | Speculative Draft Decoding  |
| 5. Token Decoding & JSON Parsing         | 5 ms     | SIMD C++ Fast JSON Parser   |
+------------------------------------------+----------+-----------------------------+
| TOTAL RUNTIME                            | 95 ms    | Async GPU Thread            |
+------------------------------------------+----------+-----------------------------+
```

---

## 5. Scene Descriptor Generator (`scene_descriptor.py`)

The `SceneDescriptor` module converts raw multi-modal inputs—multi-camera images, vehicle state telemetry, navigation goals, and the Layer 2 JEPA spatial hazard energy map—into an optimized textual context prompt.

```
+-----------------------------------------------------------------------------------+
|                        SCENE DESCRIPTOR GENERATION FLOW                           |
+-----------------------------------------------------------------------------------+
| Multi-Camera RGB Streams ----> [ Visual Token Extractor ]                         |
|                                          |                                        |
| JEPA Latent Energy Map  -----> [ Hazard Peak Detector ] --> Spatial Coordinates   |
|                                          |                                        |
| Vehicle Telemetry (v, yaw) --> [ Kinematics Formatter ]                           |
|                                          |                                        |
| Mission Plan & Route    -----> [ Navigation Injector  ]                           |
|                                          |                                        |
|                                          v                                        |
|                         [ Structured Prompt Builder ]                             |
|                                          |                                        |
|                                          v                                        |
|                  Combined Vision-Language Context Prompt String                   |
+-----------------------------------------------------------------------------------+
```

### 5.1 Conversion Logic & Prompt Structure
1. **JEPA Energy Feature Extraction:** Isolates spatial coordinates $(x, y)$ where JEPA energy $\mathcal{E}_{\text{JEPA}}(x, y) > \tau_{\text{energy\_threshold}}$. Computes high-hazard bounding polygons in ego-centric frame coordinates.
2. **Kinematic Context Injection:** Formats current forward speed $v_x$, lateral offset $d_{\text{lane}}$, and steering dynamics.
3. **Task Prompt Synthesis:** Merges text prompts using template schemas.

### 5.2 Concrete Example Prompts

#### Example Prompt A: Military Hand Signal Scenario
```text
[SYSTEM CONTEXT: You are the Alpamayo VLA Driving Reasoner for OMNIDRIVE Layer 4. Analyze visual frames and JEPA hazard metrics to select a driving intent.]
[VEHICLE TELEMETRY: Speed=42.5 km/h, SteeringAngle=0.01 rad, LaneOffset=+0.05m]
[JEPA HAZARD ENERGY: High anomaly energy peak (E=0.88) detected at x=+14.2m, y=0.0m along lane axis.]
[VISUAL OBSERVATION: Forward camera shows a dismounted person wearing military camouflage uniform standing in the center of the lane. Right arm is raised vertically with palm facing vehicle in a universal STOP gesture. Road is clear behind the personnel.]
[NAVIGATION GOAL: Maintain forward lane trajectory toward Checkpoint Alpha.]
[TASK: Evaluate scenario, generate Chain-of-Thought reasoning, and output JSON intent containing primary_intent, standoff_distance_m, target_velocity_kmh, and corridor_bias_m.]
```

#### Example Prompt B: Flooded Road Scenario
```text
[SYSTEM CONTEXT: You are the Alpamayo VLA Driving Reasoner for OMNIDRIVE Layer 4.]
[VEHICLE TELEMETRY: Speed=35.0 km/h, SteeringAngle=-0.03 rad, RainSensor=HEAVY]
[JEPA HAZARD ENERGY: Anomaly energy peak (E=0.94) at x=+22.0m across full road width. LiDAR depth return variance is zero due to specular water surface reflection.]
[VISUAL OBSERVATION: Road surface is submerged under muddy water approximately 30cm deep. Curb lines are obscured. An unpaved gravel bypass road is visible on the right at x=+12.0m, angle=+35 deg.]
[NAVIGATION GOAL: Proceed to Forward Supply Base.]
[TASK: Evaluate scenario, generate Chain-of-Thought reasoning, and output JSON intent containing primary_intent, standoff_distance_m, target_velocity_kmh, and corridor_bias_m.]
```

---

## 6. VLA Reasoner Engine (`vla_reasoner.py`)

The `VLAReasoner` module acts as the core orchestrator within Layer 4. It coordinates input acquisition from `SceneDescriptor`, manages the execution of `AlpamayoClient`, enforces temporal consistency, and passes outputs to downstream parsers.

```
+-----------------------------------------------------------------------------------+
|                             VLA REASONER PIPELINE FLOW                            |
+-----------------------------------------------------------------------------------+
| Trigger Signal (ScenarioClassifier)                                               |
|       |                                                                           |
|       v                                                                           |
| [ VLAReasoner.step() ]                                                            |
|       |                                                                           |
|       +---> Fetch Scene Prompt from SceneDescriptor                               |
|       +---> Fetch Visual Tokens from Image Pipeline                               |
|       |                                                                           |
|       v                                                                           |
| [ Execute AlpamayoClient.infer() ]                                                |
|       |                                                                           |
|       v                                                                           |
| [ Temporal Hysteresis & Intent Memory Filter ]                                   |
|   (Prevents intent chatter over 5-frame moving window)                            |
|       |                                                                           |
|       v                                                                           |
| Returns Validated ReasonerResult (CoT + Intent + Action Tokens)                   |
+-----------------------------------------------------------------------------------+
```

### 6.1 Temporal Consistency & Anti-Flicker Logic
Because autoregressive VLA models may produce slight variations in intent wording across consecutive cycles, the `VLAReasoner` maintains a **5-frame Sliding Temporal Window Memory**. An intent transition (e.g., from `PROCEED` to `STOP`) requires a minimum confidence score $C \ge 0.85$ or 2 consecutive agreement cycles before switching state, preventing high-frequency control chatter in Layer 3.

---

## 7. Rare Scenario Handler (`rare_scenario_handler.py`)

When a rare scenario is identified, `RareScenarioHandler` dispatches processing to dedicated domain logic modules.

```
+-----------------------------------------------------------------------------------+
|                           RARE SCENARIO ROUTING MATRIX                            |
+-----------------------------------------------------------------------------------+
| Scenario Classifier Trigger                                                       |
|       |                                                                           |
|       v                                                                           |
| [ RareScenarioHandler Router ]                                                    |
|       |                                                                           |
|       +---> Class 1: HUMAN_CONTROLLED_TRAFFIC ---> Flagger / Officer Handler      |
|       +---> Class 2: ENVIRONMENTAL_HAZARD   ---> Flood / Debris Reroute Handler   |
|       +---> Class 3: PHYSICAL_OBSTRUCTION     ---> Barrier Standoff Handler       |
|       +---> Class 4: TACTICAL_PROCEDURAL      ---> Military Checkpoint Handler    |
|       +---> Class 5: SPECIAL_VEHICLE         ---> Emergency Vehicle Yield Handler |
|       +---> Class 6: HEAVY_TRUCK_RESTRICTION  ---> Bridge Height / Weigh Station  |
+-----------------------------------------------------------------------------------+
```

---

## 8. Scenario Classifier (`scenario_classifier.py`)

To prevent continuous GPU resource consumption, Alpamayo VLA reasoning is invoked conditionally by the `ScenarioClassifier`. This lightweight machine learning classifier operates synchronously in 3ms, evaluating whether the current state exceeds OOD uncertainty thresholds.

```
+-----------------------------------------------------------------------------------+
|                         SCENARIO CLASSIFIER TRIGGER LOGIC                         |
+-----------------------------------------------------------------------------------+
| Layer 2 JEPA Hazard Map (Spatial Energy E_jepa)                                   |
| Layer 3 RL Policy Entropy / Confidence (C_rl)                                     |
| Perception Object Classifiers (OOD Detection Vector)                              |
|       |                                                                           |
|       v                                                                           |
| [ Trigger Function Evaluation ]                                                   |
|                                                                                   |
|        Trigger = ( E_jepa > Tau_hazard ) AND ( C_rl < Tau_confidence )            |
|                                                                                   |
|       +-----------------------------------+-----------------------------------+   |
|       | TRUE: Trigger Alpamayo VLA Loop   | FALSE: Stay in RL Nominal Mode    |   |
|       +-----------------------------------+-----------------------------------+   |
+-----------------------------------------------------------------------------------+
```

### 8.1 Mathematical Trigger Formulation

The trigger decision variable $T \in \{0, 1\}$ is defined as:

$$T = \mathbb{I}\left( \max_{(x,y)} \mathcal{E}_{\text{JEPA}}(x, y) > \tau_{\text{hazard}} \right) \lor \mathbb{I}\left( \mathcal{H}(\pi_{\text{RL}}(a|s)) > \tau_{\text{entropy}} \right) \lor \mathbb{I}\left( S_{\text{OOD}} > \tau_{\text{ood}} \right)$$

Where:
- $\mathcal{E}_{\text{JEPA}}(x, y)$ is the normalized spatial hazard energy from Layer 2.
- $\mathcal{H}(\pi_{\text{RL}}(a|s)) = -\int \pi_{\text{RL}}(a|s) \log \pi_{\text{RL}}(a|s) \, da$ is the action distribution entropy of the Layer 3 RL policy.
- $S_{\text{OOD}}$ is the out-of-distribution feature distance from the perception encoder.
- Threshold values (default): $\tau_{\text{hazard}} = 0.75$, $\tau_{\text{entropy}} = 1.85 \text{ nats}$, $\tau_{\text{ood}} = 0.80$.

---

## 9. Intent Parser (`intent_parser.py`)

The `IntentParser` converts raw text and discrete action tokens generated by Alpamayo into strongly-typed, deterministic C++/Python data structures.

```
+-----------------------------------------------------------------------------------+
|                            INTENT PARSING REGIME                                  |
+-----------------------------------------------------------------------------------+
| Alpamayo Output Payload (Text + Action Tokens)                                    |
|       |                                                                           |
|       v                                                                           |
| [ Schema Validation & Regex Extraction ]                                          |
|       |                                                                           |
|       +---> Extract JSON Block via Regex Regex pattern `\{.*?\}`                  |
|       +---> Validate against Pydantic / C++ Struct Schema                         |
|       +---> Fallback: Lexical Keyword Parser (if JSON is malformed)               |
|       |                                                                           |
|       v                                                                           |
| Structured DrivingIntent Object                                                   |
|   - Enum: {STOP, SLOW_DOWN, TURN_LEFT, TURN_RIGHT, REROUTE, YIELD, PROCEED}       |
|   - Floating point constraints: standoff_m, target_speed_kmh, bias_m              |
+-----------------------------------------------------------------------------------+
```

### 9.1 Primitive Intent Definitions

```
+-----------------------------------------------------------------------------------+
| ENUM VALUE    | DESCRIPTION & ACTION PARAMETERS                                   |
+---------------+-------------------------------------------------------------------+
| STOP          | Complete deceleration to 0 km/h at specified standoff distance.   |
| SLOW_DOWN     | Decelerate to target speed limit (e.g., 10-20 km/h).               |
| TURN_LEFT     | Execute left turn trajectory at specified intersection / junction.|
| TURN_RIGHT    | Execute right turn trajectory at specified intersection / junction.|
| REROUTE       | Abandon current lane/path; initiate global detour path planner.   |
| YIELD         | Pause trajectory, monitor object, proceed only when clear.        |
| PROCEED       | Maintain standard navigation trajectory and normal speed limit.   |
+-----------------------------------------------------------------------------------+
```

---

## 10. Command Translator (`command_translator.py`)

The `CommandTranslator` converts high-level `DrivingIntent` objects into `RLControllerHint` payloads that directly modify the behavior of the Layer 3 RL Motion Controller.

```
+-----------------------------------------------------------------------------------+
|                      COMMAND TRANSLATION & HINT INJECTION                         |
+-----------------------------------------------------------------------------------+
| Structured DrivingIntent Object                                                   |
|       |                                                                           |
|       v                                                                           |
| [ CommandTranslator Engine ]                                                      |
|       |                                                                           |
|       +---> Compute Reward Mask Modifier: R_hint(s, a)                            |
|       +---> Compute Action Bounds: [a_min, a_max]                                 |
|       +---> Compute Speed Target Penalty Vector: (v_ref - v_current)^2            |
|       |                                                                           |
|       v                                                                           |
| RLControllerHint Output Packet                                                    |
|   - inject_reward_weight: float [0.0, 1.0]                                        |
|   - velocity_cap_mps: float                                                       |
|   - action_mask_acceleration: (min_accel, max_accel)                              |
|   - lateral_bias_meters: float                                                    |
+-----------------------------------------------------------------------------------+
```

### 10.1 Hint Injection Mathematical Formulation

Layer 3 optimizes the total reward function $R_{\text{total}}(s, a)$:

$$R_{\text{total}}(s, a) = R_{\text{nominal}}(s, a) + w_{\text{hint}} \cdot R_{\text{hint}}(s, a; \text{Intent})$$

Where $R_{\text{hint}}(s, a)$ enforces intent-specific penalties:

1. **For `STOP` Intent:**
   $$R_{\text{hint}}(s, a) = -\alpha \cdot v^2 - \beta \cdot \max(0, d_{\text{standoff}} - d_{\text{current}})^2$$
   Action space bounds dynamically clipped: $a_{\text{throttle}} \in [-5.0 \text{ m/s}^2, 0.0 \text{ m/s}^2]$.

2. **For `REROUTE` Intent:**
   $$R_{\text{hint}}(s, a) = -\gamma \cdot \| \mathbf{p}_{\text{vehicle}} - \mathbf{p}_{\text{detour}} \|^2$$

---

## 11. Military Scenario Handling Specifications

Tactical and military autonomous operations require explicit multi-stage procedural protocols and visual cue interpretation.

```
+-----------------------------------------------------------------------------------+
|                           MILITARY SCENARIO STATE MACHINES                        |
+-----------------------------------------------------------------------------------+
| 1. IED / Explosive Hazard Visual Detection Sequence:                              |
|    [ Visual Anomaly ] -> [ Enforce 30m Standoff ] -> [ Halt ] -> [ Tactical Reverse]|
|                                                                                   |
| 2. Dismounted Soldier Hand Signal Protocol:                                       |
|    [ Detect Uniform ] -> [ Parse Gesture Vector ] -> [ Override Traffic Signal ]  |
|                                                                                   |
| 3. Convoy Merge & Tactical Interval Maintenance:                                  |
|    [ Identify Convoy Flag ] -> [ Maintain 50m Standoff ] -> [ Execute Merge ]     |
|                                                                                   |
| 4. Multi-Stage Military Checkpoint Procedure:                                     |
|    Approach (30 km/h) -> Standoff Line (0 km/h) -> Wait Guard Signal -> Proceed    |
+-----------------------------------------------------------------------------------+
```

### 11.1 Military Scenario Execution Matrix

```
+-----------------------------------------------------------------------------------+
| SCENARIO NAME     | VISUAL CUE SIGNATURE              | ACTION PROTOCOL            |
+-------------------+-----------------------------------+----------------------------+
| IED Detection     | Disturbed soil pattern, wires,    | Issue STOP at 30m standoff.|
|                   | suspicious roadside trash mound.  | Log GPS coordinates.       |
|                   |                                   | Trigger tactical reverse.  |
+-------------------+-----------------------------------+----------------------------+
| Soldier Hand      | Raised palm, horizontal arm wave, | Hand signal overrides all  |
| Signals           | night torch signals.              | static signals/lights.     |
|                   |                                   | Set velocity_cap = 0 km/h. |
+-------------------+-----------------------------------+----------------------------+
| Convoy Merge      | Lead vehicle amber beacon, rear   | Lock follow distance at    |
|                   | convoy flag marker.               | exactly 50m (+-2m).        |
+-------------------+-----------------------------------+----------------------------+
| Checkpoint        | Guard booth, barrier gate, armed  | Stage 1: Slow to 10 km/h.  |
| Protocol          | guard gesture.                    | Stage 2: Stop at line.     |
|                   |                                   | Stage 3: Await green wave. |
|                   |                                   | Stage 4: Proceed at 5km/h. |
+-------------------+-----------------------------------+----------------------------+
```

---

## 12. Heavy Truck & Commercial Vehicle Scenario Specifications

Commercial heavy trucks (Class 8 tractor-trailers) possess severe kinematic constraints, clearance limitations, and regulatory requirements.

```
+-----------------------------------------------------------------------------------+
|                        HEAVY TRUCK SCENARIO SPECIFICATIONS                        |
+-----------------------------------------------------------------------------------+
| 1. Low Bridge & Height Clearance Scanning:                                        |
|    - Parses clearance signs (e.g., "12'-6\" CLEARANCE").                          |
|    - Compares sign value against vehicle height profile (e.g., 4.1m).            |
|    - Triggers REROUTE intent if clearance < vehicle_height + 0.15m margin.        |
|                                                                                   |
| 2. Weigh Station Bypass / Entry Logic:                                            |
|    - Parses roadside VMS displays ("WEIGH STATION OPEN - ALL TRUCKS MUST ENTER"). |
|    - Issues TURN_RIGHT / EXIT intent 500m prior to weigh station ramp.            |
|                                                                                   |
| 3. Loading Dock Reverse Approach:                                                 |
|    - Tracks visual dock bumper guidelines and trailer articulation angle.         |
|    - Sets speed cap to 2.5 km/h with high precision lateral offset control.       |
|                                                                                   |
| 4. Restricted Roadway / No-Truck Zone Enforcement:                                |
|    - Detects sign symbols ("NO TRUCKS", "MAX WEIGHT 10T").                        |
|    - Initiates immediate turn-around or detour intent.                            |
+-----------------------------------------------------------------------------------+
```

---

## 13. Asynchronous Multi-Threaded Execution Architecture

To preserve real-time safety, Layer 3 (RL Motion Controller) executes strictly on a **12ms (83.3 Hz)** hard deadline. The Alpamayo VLA model, which requires **~100ms** per forward pass, operates asynchronously on a dedicated GPU thread using a **Lock-Free Double Buffer Shared Memory Architecture**.

```
+-----------------------------------------------------------------------------------+
|                 ASYNC THREADING & DOUBLE-BUFFER SYNCHRONIZATION                   |
+-----------------------------------------------------------------------------------+
| LAYER 4: VLA REASONER THREAD (GPU Stream 2) - 100ms Cadence (10 Hz)               |
|                                                                                   |
|  [ Read Camera & JEPA ] -> [ Alpamayo Forward Pass ] -> [ Intent Parser ]         |
|                                                               |                   |
|                                                               v                   |
|                                                  +------------------------+       |
|                                                  | Atomic Pointer Swap    |       |
|                                                  +------------------------+       |
|                                                               |                   |
|                                                               v                   |
|  SHARED DOUBLE BUFFER:                       [ Write Buffer ] <-> [ Read Buffer ] |
|                                                                     ^             |
|                                                                     |             |
| LAYER 3: RL MOTION CONTROLLER THREAD (GPU Stream 1) - 12ms Loop     |             |
|                                                                     |             |
|  [ Tick 83Hz ] ---------------> Read Latest RLControllerHint ------+             |
|  (Non-blocking atomic read; never waits for Layer 4 GPU execution)                |
+-----------------------------------------------------------------------------------+
```

### 13.1 Thread Safety & Lock-Free State Synchronization
- **Atomic Pointer Swapping:** The shared state structure `RLControllerHint` is stored in a double buffer. The VLA thread writes to the secondary buffer and performs an atomic exchange of the read pointer (`std::atomic<RLControllerHint*>`).
- **Hint Expiration & Decay:** If Layer 4 fails to issue an updated hint within 500ms (5 missing cycles), Layer 3 exponentially decays $w_{\text{hint}} \to 0$, smoothly reverting to standard nominal RL control.

---

## 14. Integration Architecture & System Dataflow

```
+-----------------------------------------------------------------------------------+
|                        FULL SYSTEM DATAFLOW & INTERFACES                          |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +---------------------------------+      +----------------------------------+    |
|  | Layer 1: Sensor Perception      |      | Layer 2: JEPA World Model        |    |
|  | (Camera Frames, Telemetry)      |      | (Hazard Energy Map E_jepa)       |    |
|  +---------------------------------+      +----------------------------------+    |
|                  |                                          |                     |
|                  +--------------------+---------------------+                     |
|                                       |                                           |
|                                       v                                           |
|                     +-----------------------------------+                         |
|                     | ScenarioClassifier.check_trigger()|                         |
|                     +-----------------------------------+                         |
|                                       |                                           |
|                             [ Trigger == TRUE ]                                   |
|                                       |                                           |
|                                       v                                           |
|                     +-----------------------------------+                         |
|                     | SceneDescriptor.generate_prompt() |                         |
|                     +-----------------------------------+                         |
|                                       |                                           |
|                                       v                                           |
|                     +-----------------------------------+                         |
|                     | AlpamayoClient.infer_async()      |  <-- Layer 4            |
|                     +-----------------------------------+     Async Thread        |
|                                       |                       (~100ms)            |
|                                       v                                           |
|                     +-----------------------------------+                         |
|                     | IntentParser.parse_payload()      |                         |
|                     +-----------------------------------+                         |
|                                       |                                           |
|                                       v                                           |
|                     +-----------------------------------+                         |
|                     | CommandTranslator.translate()     |                         |
|                     +-----------------------------------+                         |
|                                       |                                           |
|                                       v                                           |
|                         (RLControllerHint Packet)                                 |
|                                       |                                           |
|                                       v                                           |
|                     +-----------------------------------+                         |
|                     | Atomic Double-Buffer Swap         |                         |
|                     +-----------------------------------+                         |
|                                       |                                           |
|                                       v                                           |
|  +-----------------------------------------------------------------------------+  |
|  | Layer 3: RL Motion Controller Loop (12ms / 83.3 Hz Hard Real-Time Thread)   |  |
|  | Reads RLControllerHint -> Modifies Reward Surface & Action Bounds          |  |
|  +-----------------------------------------------------------------------------+  |
|                                       |                                           |
|                                       v                                           |
|  +-----------------------------------------------------------------------------+  |
|  | Layer 5: Safety Shield & Barrier Functions (Deterministic Hard Override)    |  |
|  +-----------------------------------------------------------------------------+  |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 15. Configuration Schema

The configuration parameters for the Reasoning Module are managed via a centralized YAML configuration file (`reasoning_config.yaml`), mirrored as a strongly-typed Python dataclass.

### 15.1 Configuration YAML Example (`reasoning_config.yaml`)

```yaml
reasoning_module:
  enabled: true
  model:
    tier: "alpamayo_2_super" # Options: alpamayo_1_base, alpamayo_2_super
    huggingface_repo: "nvidia/alpamayo-2-70b-super"
    engine_path: "/opt/omnidrive/models/alpamayo_2_super_fp8.plan"
    precision: "fp8" # Options: fp16, fp8, int8
    tensor_parallel_size: 2
    max_context_length: 8192
    speculative_decoding: true

  trigger_thresholds:
    jepa_hazard_energy_min: 0.75
    rl_confidence_max: 0.35
    rl_entropy_min: 1.85
    ood_score_min: 0.80

  async_execution:
    target_latency_ms: 100
    timeout_ms: 200
    sliding_memory_window_size: 5
    hint_decay_halflife_ms: 250

  military_modes:
    enable_ied_detection: true
    enable_checkpoint_protocol: true
    checkpoint_standoff_meters: 25.0

  truck_modes:
    enable_height_check: true
    vehicle_height_meters: 4.15
    vehicle_weight_tons: 36.0
```

---

## 16. API Interface & Python Class Stubs

Below are the complete, production-grade Python interface stubs for all components of the Reasoning Module.

```python
"""
OMNIDRIVE Layer 4: Reasoning Module API Specification
NVIDIA Alpamayo Vision-Language-Action Integration
"""

import enum
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class DrivingIntentEnum(enum.Enum):
    STOP = "STOP"
    SLOW_DOWN = "SLOW_DOWN"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    REROUTE = "REROUTE"
    YIELD = "YIELD"
    PROCEED = "PROCEED"


class RareScenarioCategory(enum.Enum):
    HUMAN_CONTROLLED_TRAFFIC = "HUMAN_CONTROLLED_TRAFFIC"
    ENVIRONMENTAL_HAZARD = "ENVIRONMENTAL_HAZARD"
    PHYSICAL_OBSTRUCTION = "PHYSICAL_OBSTRUCTION"
    TACTICAL_PROCEDURAL = "TACTICAL_PROCEDURAL"
    SPECIAL_VEHICLE_INTERACTION = "SPECIAL_VEHICLE_INTERACTION"
    HEAVY_TRUCK_RESTRICTION = "HEAVY_TRUCK_RESTRICTION"
    NONE = "NONE"


@dataclass
class VehicleTelemetry:
    speed_mps: float
    steering_angle_rad: float
    yaw_rate_radps: float
    acceleration_mps2: float
    current_lane_id: str
    timestamp_ns: int


@dataclass
class DrivingIntent:
    primary_intent: DrivingIntentEnum
    confidence: float
    standoff_distance_m: float
    target_velocity_mps: float
    corridor_bias_m: float
    chain_of_thought: str
    scenario_category: RareScenarioCategory
    timestamp_ns: int


@dataclass
class RLControllerHint:
    inject_reward_weight: float
    velocity_cap_mps: float
    action_mask_accel_min: float
    action_mask_accel_max: float
    lateral_bias_meters: float
    target_intent: DrivingIntentEnum
    timestamp_ns: int
    is_valid: bool


class ScenarioClassifier:
    """Evaluates whether current state warrants invoking Layer 4 Alpamayo reasoning."""

    def __init__(self, hazard_threshold: float = 0.75, entropy_threshold: float = 1.85):
        self.hazard_threshold = hazard_threshold
        self.entropy_threshold = entropy_threshold

    def evaluate(self, jepa_hazard_map: np.ndarray, rl_entropy: float, ood_score: float) -> bool:
        max_hazard = float(np.max(jepa_hazard_map)) if jepa_hazard_map.size > 0 else 0.0
        trigger = (max_hazard > self.hazard_threshold) or (rl_entropy > self.entropy_threshold) or (ood_score > 0.80)
        return trigger


class SceneDescriptor:
    """Converts multi-camera images, JEPA hazard map, and telemetry into text prompt."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_prompt(
        self,
        camera_frames: List[np.ndarray],
        jepa_hazard_map: np.ndarray,
        telemetry: VehicleTelemetry,
        navigation_goal: str
    ) -> str:
        max_energy = float(np.max(jepa_hazard_map)) if jepa_hazard_map.size > 0 else 0.0
        prompt = (
            f"[SYSTEM CONTEXT: You are Alpamayo VLA for OMNIDRIVE Layer 4.]\n"
            f"[VEHICLE TELEMETRY: Speed={telemetry.speed_mps * 3.6:.1f} km/h, Steering={telemetry.steering_angle_rad:.2f} rad]\n"
            f"[JEPA HAZARD MAP: Energy Peak E={max_energy:.2f}]\n"
            f"[NAVIGATION GOAL: {navigation_goal}]\n"
            f"[TASK: Output Chain-of-Thought and JSON intent containing primary_intent, standoff_distance_m, target_velocity_kmh.]"
        )
        return prompt


class AlpamayoClient:
    """Loads and executes NVIDIA Alpamayo VLA model locally via TensorRT-LLM engine."""

    def __init__(self, engine_path: str, model_tier: str = "alpamayo_2_super"):
        self.engine_path = engine_path
        self.model_tier = model_tier
        self.is_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        # Stub for TensorRT-LLM / vLLM model engine loading & CUDA stream creation
        self.is_loaded = True

    def infer(self, prompt_text: str, camera_patches: np.ndarray) -> str:
        """Executes sync/async inference returning raw text response (CoT + JSON block)."""
        if not self.is_loaded:
            raise RuntimeError("Alpamayo model engine is not loaded.")
        
        # Synthetic fallback response for API stub demonstration
        mock_response = (
            "THOUGHT: Detected dismounted soldier waving palm to stop at +15m.\n"
            '```json\n'
            '{\n'
            '  "primary_intent": "STOP",\n'
            '  "confidence": 0.96,\n'
            '  "standoff_distance_m": 15.0,\n'
            '  "target_velocity_kmh": 0.0,\n'
            '  "corridor_bias_m": 0.0,\n'
            '  "scenario_category": "TACTICAL_PROCEDURAL"\n'
            '}\n'
            '```'
        )
        return mock_response


class IntentParser:
    """Parses raw text/tokens from Alpamayo into structured DrivingIntent object."""

    def parse(self, raw_output: str) -> DrivingIntent:
        import re, json
        json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return DrivingIntent(
                primary_intent=DrivingIntentEnum(data.get("primary_intent", "STOP")),
                confidence=float(data.get("confidence", 0.9)),
                standoff_distance_m=float(data.get("standoff_distance_m", 0.0)),
                target_velocity_mps=float(data.get("target_velocity_kmh", 0.0)) / 3.6,
                corridor_bias_m=float(data.get("corridor_bias_m", 0.0)),
                chain_of_thought=raw_output.split("```")[0].strip(),
                scenario_category=RareScenarioCategory(data.get("scenario_category", "NONE")),
                timestamp_ns=time.time_ns()
            )
        raise ValueError("Failed to parse valid JSON intent from Alpamayo output.")


class CommandTranslator:
    """Translates DrivingIntent into RLControllerHint for Layer 3 motion controller."""

    def translate(self, intent: DrivingIntent) -> RLControllerHint:
        if intent.primary_intent == DrivingIntentEnum.STOP:
            return RLControllerHint(
                inject_reward_weight=1.0,
                velocity_cap_mps=0.0,
                action_mask_accel_min=-5.0,
                action_mask_accel_max=0.0,
                lateral_bias_meters=0.0,
                target_intent=intent.primary_intent,
                timestamp_ns=time.time_ns(),
                is_valid=True
            )
        elif intent.primary_intent == DrivingIntentEnum.SLOW_DOWN:
            return RLControllerHint(
                inject_reward_weight=0.8,
                velocity_cap_mps=intent.target_velocity_mps,
                action_mask_accel_min=-3.0,
                action_mask_accel_max=1.0,
                lateral_bias_meters=intent.corridor_bias_m,
                target_intent=intent.primary_intent,
                timestamp_ns=time.time_ns(),
                is_valid=True
            )
        else:
            return RLControllerHint(
                inject_reward_weight=0.0,
                velocity_cap_mps=30.0,
                action_mask_accel_min=-5.0,
                action_mask_accel_max=3.0,
                lateral_bias_meters=0.0,
                target_intent=DrivingIntentEnum.PROCEED,
                timestamp_ns=time.time_ns(),
                is_valid=True
            )


class VLAReasoner:
    """Master Orchestrator for Layer 4 Reasoning Module."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.classifier = ScenarioClassifier()
        self.descriptor = SceneDescriptor(config)
        self.client = AlpamayoClient(config["model"]["engine_path"])
        self.parser = IntentParser()
        self.translator = CommandTranslator()

    def process_step(
        self,
        camera_frames: List[np.ndarray],
        jepa_map: np.ndarray,
        rl_entropy: float,
        telemetry: VehicleTelemetry,
        nav_goal: str
    ) -> Optional[RLControllerHint]:
        # Step 1: Check trigger
        triggered = self.classifier.evaluate(jepa_map, rl_entropy, ood_score=0.85)
        if not triggered:
            return None

        # Step 2: Generate scene prompt
        prompt = self.descriptor.generate_prompt(camera_frames, jepa_map, telemetry, nav_goal)

        # Step 3: Run Alpamayo VLA inference
        raw_output = self.client.infer(prompt, camera_patches=np.array([]))

        # Step 4: Parse intent
        intent = self.parser.parse(raw_output)

        # Step 5: Translate to RL Hint
        hint = self.translator.translate(intent)
        return hint
```

---

## 17. Unit Test Plan & Verification Suite

To ensure absolute safety and reliability before hardware-in-the-loop (HIL) deployment, the Reasoning Module undergoes rigorous automated unit and integration testing.

```
+-----------------------------------------------------------------------------------+
|                        UNIT TEST SUITE SUMMARY MATRIX                             |
+-----------------------------------------------------------------------------------+
| TEST CASE ID | TEST SCENARIO DESCRIPTION          | EXPECTED SYSTEM OUTPUT        |
+--------------+------------------------------------+-------------------------------+
| **TC-R01**   | Flooded Road Submersion (30cm)     | Intent: REROUTE               |
|              |                                    | Corridor Bias: +4.5m          |
|              |                                    | Velocity Cap: 10 km/h         |
+--------------+------------------------------------+-------------------------------+
| **TC-R02**   | Soldier Stop Gesture at Checkpoint | Intent: STOP                  |
|              |                                    | Standoff: 15.0m               |
|              |                                    | Velocity Cap: 0 km/h          |
+--------------+------------------------------------+-------------------------------+
| **TC-R03**   | Low Bridge (3.5m height sign)      | Intent: REROUTE               |
|              | (Truck Height = 4.15m)             | Trigger Detour Path Planner   |
+--------------+------------------------------------+-------------------------------+
| **TC-R04**   | High JEPA Hazard Energy (>0.90)    | Trigger = TRUE                |
|              | & Low RL Confidence (<0.20)        | Alpamayo Client Invoked       |
+--------------+------------------------------------+-------------------------------+
| **TC-R05**   | TensorRT Execution Latency         | Latency <= 98ms               |
|              | Benchmark (<100ms Budget)          | Zero GPU memory leaks         |
+--------------+------------------------------------+-------------------------------+
| **TC-R06**   | Lock-Free Buffer Race Condition    | 1,000,000 Atomic Reads        |
|              | Concurrent Read/Write (83Hz vs 10Hz)| Zero Data Corruption / Stalls|
+--------------+------------------------------------+-------------------------------+
```

---

## 18. Appendix: Example Input/Output JSON Schemas & Payloads

### 18.1 Alpamayo Request Payload JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AlpamayoInferenceRequest",
  "type": "object",
  "properties": {
    "request_id": { "type": "string" },
    "timestamp_ns": { "type": "integer" },
    "telemetry": {
      "type": "object",
      "properties": {
        "speed_mps": { "type": "number" },
        "steering_angle_rad": { "type": "number" },
        "yaw_rate_radps": { "type": "number" }
      },
      "required": ["speed_mps", "steering_angle_rad"]
    },
    "jepa_metrics": {
      "type": "object",
      "properties": {
        "max_hazard_energy": { "type": "number" },
        "peak_coordinate_ego": {
          "type": "array",
          "items": { "type": "number" },
          "minItems": 2,
          "maxItems": 2
        }
      },
      "required": ["max_hazard_energy"]
    },
    "prompt_text": { "type": "string" }
  },
  "required": ["request_id", "timestamp_ns", "telemetry", "prompt_text"]
}
```

### 18.2 Alpamayo Response Payload JSON Example

```json
{
  "response_id": "alp-resp-20260810-00421",
  "latency_ms": 91.4,
  "chain_of_thought": "Forward visual camera detects dismounted military personnel standing in ego lane with palm raised vertically. JEPA hazard energy peaks at E=0.89 at +14.5 meters. Standard traffic lights are not present. Hand signal overrides nominal speed profile.",
  "driving_intent": {
    "primary_intent": "STOP",
    "confidence": 0.985,
    "standoff_distance_m": 14.5,
    "target_velocity_kmh": 0.0,
    "corridor_bias_m": 0.0,
    "scenario_category": "TACTICAL_PROCEDURAL"
  },
  "action_tokens": [14, 88, 201, 12, 0, 0, 4, 19],
  "rl_controller_hint": {
    "inject_reward_weight": 1.0,
    "velocity_cap_mps": 0.0,
    "action_mask_accel_min": -5.0,
    "action_mask_accel_max": 0.0,
    "lateral_bias_meters": 0.0,
    "is_valid": true
  }
}
```

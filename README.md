# Vision-Language Panda Pick-and-Place

A MuJoCo-based Vision-Language-Action robotics project for natural-language-controlled pick-and-place with a Franka Emika Panda robot.

This project demonstrates how a natural-language command such as:

```text
pick the red cube and place it at x 0.55 y -0.45
```

is converted into:

```text
language command
→ parsed task
→ object grounding
→ symbolic action
→ numerical action vector
→ robot action sequence
→ MuJoCo robot execution
→ demonstration dataset
→ behavior cloning
→ diffusion-style action prediction
→ world-model next-state prediction
→ preference-based action ranking
→ RL-style reward evaluation
→ video-action dataset
```

The goal is to build a compact end-to-end prototype connecting language grounding, action encoding, robot manipulation, demonstration learning, and research-style embodied AI components.

---

## Demo

The system supports natural-language commands for colored cube manipulation in MuJoCo.

Example commands:

```text
pick the red cube and place it at x 0.55 y -0.45
move the yellow block to x 0.45 y 0.20
put the green cube to the right
pick the blue cube and place it in the bin
put the red cube to the left
```

Example VLA output:

```text
[LANGUAGE COMMAND]
pick the red cube and place it at x 0.55 y -0.45

[PARSED COMMAND]
{'task': 'pick_place_xy', 'obj': 'red_box', 'x': 0.55, 'y': -0.45}

[OBJECT GROUNDING]
{'object_name': 'red_box', 'object_position': [0.4, -0.3, 0.03], 'visible_or_known': True}

[SYMBOLIC ACTION]
{'action_type': 'pick_place_xy', 'object_name': 'red_box', ...}

[ACTION VECTOR]
[2.0, 1.0, 0.4, -0.3, 0.55, -0.45, 0.0, 1.0]

[ACTION SEQUENCE]
01. APPROACH_OBJECT
02. OPEN_GRIPPER
03. DESCEND_TO_OBJECT
04. CLOSE_GRIPPER
05. LIFT_OBJECT
06. MOVE_TO_TARGET
07. DESCEND_TO_PLACE
08. OPEN_GRIPPER
09. RETREAT
```

---

## Key Features

- Natural-language command parsing
- Object/color grounding for red, green, blue, yellow, and default boxes
- MuJoCo simulation with Franka Emika Panda robot
- Cartesian pick-and-place execution
- VLA-style symbolic action encoder
- Numerical action vector representation
- Robot action sequence generation
- Demonstration dataset logger
- Dataset loader for behavior cloning preparation
- RandomForest behavior cloning baseline
- Nearest-neighbor behavior cloning baseline
- Tiny diffusion-style action policy prototype
- Tiny world model prototype for next-state prediction
- Preference learning prototype for action ranking
- Reinforcement-learning-style reward function
- Random policy evaluation
- Video Action Dataset / VAM-style prototype

---

## Project Structure

```text
vision-language-panda-pick-place/
│
├── pickandplace.py                     # Main MuJoCo Panda pick-and-place controller
├── nl_interface.py                     # Natural-language command parser
├── run_vla_action_demo.py              # Standalone VLA pipeline demo
│
├── action_encoding/
│   ├── __init__.py
│   └── action_encoder.py               # Symbolic and vector action encoder
│
├── action_sequence/
│   ├── __init__.py
│   └── sequence_encoder.py             # Converts actions into robot step sequences
│
├── perception/
│   ├── __init__.py
│   ├── color_detector.py               # Simple color-based object detection
│   └── vision_grounder.py              # Vision-language object grounding
│
├── dataset/
│   ├── __init__.py
│   ├── demo_logger.py                  # Saves VLA demonstrations as JSON
│   └── dataset_loader.py               # Loads demos into X/Y learning arrays
│
├── datasets/
│   └── demo_0001.json                  # Saved demonstrations
│
├── diffusion_policy/
│   ├── __init__.py
│   ├── train_tiny_diffusion_policy.py  # Tiny denoising action model
│   └── predict_tiny_diffusion_action.py
│
├── world_model/
│   ├── __init__.py
│   ├── train_tiny_world_model.py       # Next-state prediction model
│   └── predict_next_state.py
│
├── preference_learning/
│   ├── __init__.py
│   ├── create_preference_dataset.py
│   ├── train_preference_ranker.py
│   ├── rank_candidate_actions.py
│   └── preferences.json
│
├── rl/
│   ├── __init__.py
│   ├── reward_function.py
│   ├── evaluate_action_reward.py
│   └── random_policy_eval.py
│
├── video_action/
│   ├── __init__.py
│   ├── extract_video_frames.py
│   ├── create_video_action_dataset.py
│   ├── inspect_video_action_dataset.py
│   └── datasets/
│       └── video_action_dataset.json
│
├── models/
│   ├── bc_policy.pkl
│   ├── nn_bc_policy.pkl
│   ├── tiny_diffusion_policy.pt
│   ├── tiny_world_model.pkl
│   └── preference_ranker.pkl
│
├── train_bc_baseline.py
├── predict_bc_action.py
├── train_nn_bc_baseline.py
├── predict_nn_bc_action.py
│
├── test_action_encoder.py
├── test_dataset_loader.py
├── test_perception.py
├── test_vision_language.py
│
├── world.xml
├── panda.xml
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

```text
Natural Language
      ↓
Command Parser
      ↓
Object Grounding
      ↓
Symbolic Action Encoder
      ↓
Numerical Action Vector
      ↓
Robot Action Sequence
      ↓
MuJoCo Robot Execution
      ↓
Demonstration Logger
      ↓
Dataset Loader
      ↓
Learning Prototypes
```

Learning prototypes include:

```text
Behavior Cloning
Nearest-Neighbor Imitation
Diffusion-Style Action Prediction
World Model Prediction
Preference-Based Action Ranking
RL-Style Reward Evaluation
Video-Action Dataset Generation
```

---

## Action Representation

### Symbolic Action

```python
{
    "action_type": "pick_place_xy",
    "action_type_id": 2,
    "object_name": "red_box",
    "object_id": 1,
    "target_xy": [0.55, -0.45],
    "control_mode": "cartesian",
    "gripper_sequence": ["open", "close", "open"]
}
```

### Numerical Action Vector

Format:

```text
[
    action_type_id,
    object_id,
    pick_x,
    pick_y,
    place_x,
    place_y,
    target_id,
    has_explicit_xy
]
```

Example:

```text
[2.0, 1.0, 0.4, -0.3, 0.55, -0.45, 0.0, 1.0]
```

---

## Action Sequence Encoding

Each language instruction is converted into robot-executable action steps:

```text
01. APPROACH_OBJECT
02. OPEN_GRIPPER
03. DESCEND_TO_OBJECT
04. CLOSE_GRIPPER
05. LIFT_OBJECT
06. MOVE_TO_TARGET
07. DESCEND_TO_PLACE
08. OPEN_GRIPPER
09. RETREAT
```

This provides a bridge between high-level language commands and low-level robot execution.

---

## Demonstration Dataset

The project saves each VLA demonstration as JSON.

Each demo stores:

- language command
- parsed task
- object grounding
- symbolic action
- numerical action vector
- decoded action
- robot action sequence
- timestamp

Example:

```json
{
  "demo_id": 1,
  "timestamp": "2026-05-11T20:31:39",
  "data": {
    "language": "pick the red cube and place it at x 0.55 y -0.45",
    "parsed_task": {
      "task": "pick_place_xy",
      "obj": "red_box",
      "x": 0.55,
      "y": -0.45
    },
    "object_grounding": {
      "object_name": "red_box",
      "object_position": [0.4, -0.3, 0.03],
      "visible_or_known": true
    },
    "action_vector": [2.0, 1.0, 0.4, -0.3, 0.55, -0.45, 0.0, 1.0]
  }
}
```

These demonstrations can be used for:

- behavior cloning
- imitation learning
- diffusion-policy-style learning
- sequence modeling
- world-model-conditioned action prediction
- preference learning
- reward-based policy evaluation

---

## Behavior Cloning Baselines

### RandomForest Behavior Cloning

A small supervised model predicts action vectors from compact task/object features.

Input:

```text
[action_type_id, object_id, target_id, has_explicit_xy]
```

Output:

```text
[action_type_id, object_id, pick_x, pick_y, place_x, place_y, target_id, has_explicit_xy]
```

Run:

```bash
python3 train_bc_baseline.py
python3 predict_bc_action.py
```

### Nearest-Neighbor Behavior Cloning

A retrieval-based baseline finds the closest saved demonstration and reuses its action vector.

This works better for small datasets because it preserves categorical values such as object ID and target ID.

Run:

```bash
python3 train_nn_bc_baseline.py
python3 predict_nn_bc_action.py
```

Example:

```text
put the green cube to the right
→ retrieves saved green/right demonstration
→ predicts green_box + zone_right correctly
```

---

## Tiny Diffusion Policy Prototype

The project includes a lightweight diffusion-style action predictor.

It trains an MLP denoising model over action vectors:

```text
condition = [action_type_id, object_id, target_id, has_explicit_xy]
clean_action = [action_type_id, object_id, pick_x, pick_y, place_x, place_y, target_id, has_explicit_xy]
noisy_action = clean_action + Gaussian noise
model input = condition + noisy_action + noise_level
model output = denoised action vector
```

This is an action-space diffusion prototype, not yet a full image-conditioned diffusion policy.

Run:

```bash
python3 diffusion_policy/train_tiny_diffusion_policy.py
python3 diffusion_policy/predict_tiny_diffusion_action.py
```

Example output:

```text
[COMMAND]
put the green cube to the right

[POST-PROCESSED ACTION VECTOR]
[1.0, 2.0, 0.4934, -0.2936, 0.0062, -0.0022, 3.0, 0.0]

[DECODED PREDICTION]
{'action_type': 'pick_place', 'object_name': 'green_box', 'target_name': 'zone_right'}
```

---

## Tiny World Model Prototype

The project includes a lightweight world model that predicts the next object state after an action.

The model learns:

```text
current object state + action vector → predicted next object position
```

Input format:

```text
[
  object_id,
  current_x,
  current_y,
  current_z,
  action_type_id,
  pick_x,
  pick_y,
  place_x,
  place_y,
  target_id,
  has_explicit_xy
]
```

Output format:

```text
[next_x, next_y, next_z]
```

Run:

```bash
python3 world_model/train_tiny_world_model.py
python3 world_model/predict_next_state.py
```

Example:

```text
Command: pick the red cube and place it at x 0.55 y -0.45
Predicted next object position: [0.55, -0.448, 0.03]
```

---

## Preference Learning Prototype

The project includes a lightweight preference-learning prototype.

It generates synthetic preference pairs:

```text
preferred action > rejected action
```

Rejected actions are created by corrupting:

- object ID
- target ID
- placement position

The trained preference ranker scores candidate actions for a language command and ranks them.

Run:

```bash
python3 preference_learning/create_preference_dataset.py
python3 preference_learning/train_preference_ranker.py
python3 preference_learning/rank_candidate_actions.py
```

Example:

```text
Command: put the green cube to the right

Rank 1: correct_action
Decoded: {'action_type': 'pick_place', 'object_name': 'green_box', 'target_name': 'zone_right'}
```

---

## Tiny RL Reward Function Prototype

The project includes a lightweight reinforcement-learning-style reward function.

It evaluates candidate robot actions for a language command:

```text
state / command + action vector → reward
```

The reward checks:

- correct action type
- correct object
- correct target or placement location
- pick position consistency
- distance to desired placement

It also includes random policy evaluation, comparing random candidate actions against the rule-based correct action.

Run:

```bash
python3 rl/evaluate_action_reward.py
python3 rl/random_policy_eval.py
```

Example result:

```text
Correct action reward: 4.0
Average random reward: much lower
Best random reward: below correct action reward
```

This is not a full RL training loop yet, but it provides the reward and evaluation structure needed for future RL policy optimization.

---

## Video Action Dataset Prototype

The project includes a lightweight Video Action Model-style dataset prototype.

It converts a MuJoCo demo video into frame-level samples:

```text
video frame + language command + action vector + action phase label
```

Pipeline:

```text
MuJoCo demo video
→ frame extraction
→ frame-level action labels
→ video-action dataset JSON
```

Each frame sample stores:

- frame path
- language command
- parsed task
- symbolic action
- numerical action vector
- coarse action phase label
- full action sequence

Run:

```bash
python3 video_action/extract_video_frames.py --video media/vla_panda_demo.mp4 --fps 2
python3 video_action/create_video_action_dataset.py
python3 video_action/inspect_video_action_dataset.py
```

Example summary:

```text
Number of samples: 159

Action phase counts:
APPROACH_OBJECT: 19
OPEN_GRIPPER: 29
DESCEND_TO_OBJECT: 21
CLOSE_GRIPPER: 16
LIFT_OBJECT: 20
MOVE_TO_TARGET: 22
DESCEND_TO_PLACE: 19
RETREAT: 13
```

This is not a full Video Action Model yet, but it creates the dataset structure needed for future video-conditioned action prediction.

---

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install mujoco glfw numpy opencv-python SpeechRecognition pyaudio scikit-learn joblib torch
```

On Ubuntu, PyAudio may require PortAudio:

```bash
sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio
```

---

## Running the Main Robot Demo

Terminal-based natural-language control:

```bash
python3 pickandplace.py --terminal
```

Example command:

```text
pick the red cube and place it at x 0.55 y -0.45
```

Viewer-console mode:

```bash
python3 pickandplace.py --viewer-console
```

Voice mode:

```bash
python3 pickandplace.py --voice-terminal
```

---

## Running the VLA Action Demo

```bash
python3 run_vla_action_demo.py
```

This prints:

```text
language command
parsed command
object grounding
symbolic action
action vector
decoded action
action sequence
optional saved demonstration
```

---

## Dataset Logging

Run:

```bash
python3 run_vla_action_demo.py
```

After each command, save the demonstration:

```text
Save this demonstration? [y/N]: y
```

Saved files appear in:

```text
datasets/demo_0001.json
datasets/demo_0002.json
...
```

---

## Dataset Loader

```bash
python3 test_dataset_loader.py
```

Example output:

```text
Loaded 6 demos.
X shape: (6, 4)
Y shape: (6, 8)
```

---

## Current Limitations

- The current language parser is rule-based.
- The object grounding uses simple color and object-name mappings.
- The dataset is intentionally small for demonstration purposes.
- The RandomForest BC model is a proof-of-concept and can average categorical IDs.
- The nearest-neighbor BC baseline works better for small datasets but does not generalize like a learned policy.
- The diffusion policy is action-space only, not image-conditioned.
- The world model predicts simplified object-level next states.
- The preference dataset uses synthetic corrupted actions.
- The RL component is a reward/evaluation prototype, not a full training loop.
- The video-action module creates frame/action metadata but does not yet train a video model.

---

## Future Work

Planned extensions:

- Replace rule-based parsing with LLM-based command parsing
- Use camera-based object detection during live execution
- Add image features or scene embeddings to the action encoder
- Collect larger demonstration datasets
- Train classification-regression hybrid policies
- Add sequence-level behavior cloning
- Extend diffusion policy to image-conditioned action sequences
- Train a world model from real transition data
- Add learned preference models from human feedback
- Add full RL policy optimization
- Add video-conditioned action prediction
- Integrate with ROS 2 or Isaac Lab for larger-scale robot learning workflows

---

## Why This Project Matters

This project is a compact prototype of a Vision-Language-Action robotics stack. It connects natural-language instructions to grounded robot actions, action representations, demonstrations, and learning baselines.

It is designed as a stepping stone toward:

- embodied AI
- robotic foundation models
- vision-language-action systems
- imitation learning
- diffusion-policy-style robotics
- world-model-based robot reasoning
- preference learning
- reinforcement learning
- video-action modeling
- robot manipulation
- sim-to-real robotics workflows

---

## Author

**M A Hafiz**  
Robotics Simulation & Control Engineer  
GitHub: [MAHAFIZS](https://github.com/MAHAFIZS)  
Portfolio: [mahafizsourav.com](https://mahafizsourav.com)
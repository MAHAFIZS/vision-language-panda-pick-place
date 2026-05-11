# Vision-Language Panda Pick-and-Place

A MuJoCo-based Vision-Language-Action (VLA) robotics project for natural-language-controlled pick-and-place with a Franka Emika Panda robot.

This project demonstrates how a language command such as:

```text
pick the red cube and place it at x 0.55 y -0.45
is converted into:

language command
→ parsed task
→ object grounding
→ symbolic action
→ numerical action vector
→ robot action sequence
→ demonstration dataset
→ behavior cloning baseline

The goal is to build a compact end-to-end prototype connecting language grounding, action encoding, robot manipulation, and learning-ready demonstration data.

Demo

The system supports natural-language commands for colored cube manipulation in MuJoCo.

Example commands:

pick the red cube and place it at x 0.55 y -0.45
move the yellow block to x 0.45 y 0.20
put the green cube to the right
pick the blue cube and place it in the bin
put the red cube to the left

Example VLA output:

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
Key Features
Natural-language command parsing
Object/color grounding for red, green, blue, yellow, and default boxes
MuJoCo simulation with Franka Emika Panda robot
Cartesian pick-and-place execution
VLA-style symbolic action encoder
Numerical action vector representation
Robot action sequence generation
Demonstration dataset logger
Dataset loader for behavior cloning preparation
Tiny RandomForest behavior cloning baseline
Nearest-neighbor behavior cloning baseline for small datasets


Project Structure
vision-language-panda-pick-place/
│
├── pickandplace.py                  # Main MuJoCo Panda pick-and-place controller
├── nl_interface.py                  # Natural-language command parser
├── run_vla_action_demo.py           # Standalone VLA pipeline demo
│
├── action_encoding/
│   ├── __init__.py
│   └── action_encoder.py            # Symbolic and vector action encoder
│
├── action_sequence/
│   ├── __init__.py
│   └── sequence_encoder.py          # Converts actions into robot step sequences
│
├── perception/
│   ├── __init__.py
│   ├── color_detector.py            # Simple color-based object detection
│   └── vision_grounder.py           # Vision-language object grounding
│
├── dataset/
│   ├── __init__.py
│   ├── demo_logger.py               # Saves VLA demonstrations as JSON
│   └── dataset_loader.py            # Loads demos into X/Y learning arrays
│
├── datasets/
│   └── demo_0001.json               # Saved demonstrations
│
├── models/
│   ├── bc_policy.pkl                # RandomForest BC baseline
│   └── nn_bc_policy.pkl             # Nearest-neighbor BC baseline
│
├── train_bc_baseline.py             # Train tiny RandomForest BC model
├── predict_bc_action.py             # Predict action with RandomForest BC model
├── train_nn_bc_baseline.py          # Train nearest-neighbor BC model
├── predict_nn_bc_action.py          # Predict action using nearest-neighbor retrieval
│
├── test_action_encoder.py           # Test symbolic/vector action encoding
├── test_dataset_loader.py           # Test dataset loading and BC arrays
├── test_perception.py               # Test color perception
├── test_vision_language.py          # Test language + vision grounding
│
├── world.xml                        # MuJoCo scene
├── panda.xml                        # Franka Panda robot model
└── README.md

Pipeline Overview
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
Demonstration Logger
      ↓
Dataset Loader
      ↓
Behavior Cloning Baselines

Action Representation
Symbolic Action

Example:

{
    "action_type": "pick_place_xy",
    "action_type_id": 2,
    "object_name": "red_box",
    "object_id": 1,
    "target_xy": [0.55, -0.45],
    "control_mode": "cartesian",
    "gripper_sequence": ["open", "close", "open"]
}
Numerical Action Vector

Format:

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

Example:

[2.0, 1.0, 0.4, -0.3, 0.55, -0.45, 0.0, 1.0]

Action Sequence Encoding

Each language instruction is converted into robot-executable action steps:

01. APPROACH_OBJECT
02. OPEN_GRIPPER
03. DESCEND_TO_OBJECT
04. CLOSE_GRIPPER
05. LIFT_OBJECT
06. MOVE_TO_TARGET
07. DESCEND_TO_PLACE
08. OPEN_GRIPPER
09. RETREAT

This provides a bridge between high-level language commands and low-level robot execution.
Demonstration Dataset

The project can save each VLA demonstration as JSON.

Example:

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
    "action_vector": [2.0, 1.0, 0.4, -0.3, 0.55, -0.45, 0.0, 1.0],
    "action_sequence": []
  }
}

These demonstrations can later be used for:

behavior cloning
imitation learning
diffusion-policy-style learning
sequence modeling
world-model-conditioned action prediction

Behavior Cloning Baselines
1. RandomForest BC Baseline

A small supervised model predicts action vectors from compact task/object features.

Input:

[action_type_id, object_id, target_id, has_explicit_xy]

Output:

[action_type_id, object_id, pick_x, pick_y, place_x, place_y, target_id, has_explicit_xy]

This baseline demonstrates trainable action-vector prediction, but with very small datasets categorical IDs may be averaged.

2. Nearest-Neighbor BC Baseline

A retrieval-based baseline finds the closest saved demonstration and reuses its action vector.

This works better for small datasets because it preserves categorical values such as object ID and target ID.

Example:

put the green cube to the right
→ retrieves saved green/right demonstration
→ predicts green_box + zone_right correctly

nstallation

Create and activate a Python virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install mujoco glfw numpy opencv-python SpeechRecognition pyaudio scikit-learn joblib

On Ubuntu, PyAudio may require PortAudio:

sudo apt install portaudio19-dev python3-pyaudio
pip install pyaudio

Running the Main Robot Demo

Terminal-based natural-language control:

python3 pickandplace.py --terminal

Example command:

pick the red cube and place it at x 0.55 y -0.45

Viewer-console mode:

python3 pickandplace.py --viewer-console

Voice mode:

python3 pickandplace.py --voice-terminal

Running the VLA Action Demo
python3 run_vla_action_demo.py

This prints:

language command
parsed command
object grounding
symbolic action
action vector
decoded action
action sequence
optional saved demonstration

Dataset Logging

Run:

python3 run_vla_action_demo.py

After each command, save the demonstration:

Save this demonstration? [y/N]: y

Saved files appear in:

datasets/demo_0001.json
datasets/demo_0002.json
...

Dataset Loader
python3 test_dataset_loader.py

Example output:

Loaded 6 demos.
X shape: (6, 4)
Y shape: (6, 8)

Train Behavior Cloning Baseline

RandomForest baseline:

python3 train_bc_baseline.py

Predict with RandomForest BC:

python3 predict_bc_action.py

Nearest-neighbor baseline:

python3 train_nn_bc_baseline.py

Predict with nearest-neighbor BC:

python3 predict_nn_bc_action.py

Example Nearest-Neighbor Prediction

Input:

put the green cube to the right

Output:

[RETRIEVED DEMONSTRATION]
demo_0004: put the green cube to the right

[DECODED PREDICTION]
{
    "action_type": "pick_place",
    "object_name": "green_box",
    "target_name": "zone_right"
}

Example Nearest-Neighbor Prediction

Input:

put the green cube to the right

Output:

[RETRIEVED DEMONSTRATION]
demo_0004: put the green cube to the right

[DECODED PREDICTION]
{
    "action_type": "pick_place",
    "object_name": "green_box",
    "target_name": "zone_right"
}

Current Limitations
The current language parser is rule-based.
The object grounding uses simple color and object-name mappings.
The dataset is intentionally small for demonstration purposes.
The RandomForest BC model is a proof-of-concept and can average categorical IDs.
The nearest-neighbor BC baseline works better for small datasets but does not generalize like a learned policy.
The system does not yet use a large vision-language model or diffusion policy.
Future Work

Planned extensions:

Replace rule-based parsing with LLM-based command parsing
Use camera-based object detection during live execution
Add image features or scene embeddings to the action encoder
Collect larger demonstration datasets
Train classification-regression hybrid policies
Add sequence-level behavior cloning
Prototype diffusion-policy-style action sequence generation
Add video-action-model-inspired temporal prediction
Integrate with ROS 2 or Isaac Lab for larger-scale robot learning workflows
Why This Project Matters

This project is a compact prototype of a Vision-Language-Action robotics stack. It connects natural-language instructions to grounded robot actions, action representations, demonstrations, and learning baselines.

It is designed as a stepping stone toward:

embodied AI
robotic foundation models
vision-language-action systems
imitation learning
robot manipulation
sim-to-real robotics workflows

Author

M A Hafiz
Robotics Simulation & Control Engineer
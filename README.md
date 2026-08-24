Failure-Aware Robot Learning for Franka Panda Manipulation

A MuJoCo + PyTorch robot-learning project for Franka Panda manipulation that combines:

Soft Actor-Critic (SAC)
Object-relative Cartesian control
Failure-region analysis
Targeted policy retraining
Specialist-policy routing
Grasp verification
Adaptive trajectory correction
Natural-language / VLA manipulation components

PROJECT OVERVIEW

The project explores a failure-aware robot-learning workflow:

Train policy
↓
Evaluate on unseen targets
↓
Detect systematic failures
↓
Generate targeted training distribution
↓
Train specialist policy
↓
Route between general + specialist policies
↓
Execute manipulation
↓
Detect grasp failures
↓
Adapt trajectory
↓
Re-evaluate

The central idea is that robot learning should not stop after training a single policy.

Instead, the system continuously:

Evaluates the learned policy
Discovers systematic failure regions
Generates targeted training data
Trains specialized policies
Routes tasks to the appropriate policy
Verifies manipulation success
Adapts the robot trajectory when failures are detected

FINAL RESULTS

OBJECT-RELATIVE SAC PRE-GRASP

The Franka Panda learns Cartesian object-relative pre-grasp reaching using Soft Actor-Critic.

Observation:

The SAC policy receives a 26-dimensional state containing:

End-effector position
Object position
Object-relative pre-grasp goal
Goal-to-end-effector vector
7 robot joint positions
7 robot joint velocities

Action:

3-D Cartesian delta action

The action represents small Cartesian changes in the robot end-effector position.

FAILURE-AWARE REACH RESULTS

The first evaluation was performed using previously unseen object positions.

Initial unseen-object evaluation:

Original SAC:
88 / 100 success

The failure analysis showed a strong weakness in the negative-Y workspace region.

A targeted SAC specialist was therefore trained using failure-focused sampling.

Hard-region benchmark:

Original SAC: 22 / 100
Targeted SAC: 100 / 100

Improvement:

+78 percentage points

This showed that targeted retraining on the detected failure region could dramatically improve policy performance.

FAILURE-AWARE POLICY ROUTER

After training the specialist policy, a failure-aware policy router was introduced.

Object position
↓
Is y < -0.10?
↓
Yes -> Targeted SAC
No -> Original SAC
↓
Pre-Grasp

Policy selection logic:

if object_y < -0.10:
use targeted SAC
else:
use original SAC

Hybrid unseen-object evaluation:

100 / 100 pre-grasp success

Therefore, combining the general SAC policy with the targeted specialist eliminated the detected pre-grasp failures in the evaluation benchmark.

FAILURE-AWARE GRASP PIPELINE

The learned pre-grasp policy was integrated into a complete manipulation pipeline.

Object position
↓
Failure-aware SAC router
↓
Learned pre-grasp
↓
Precision XY alignment
↓
Vertical descent
↓
Close gripper
↓
Lift
↓
Verify object motion

The system separates the learned reaching component from the precision grasp execution stages.

INITIAL FULL MANIPULATION BENCHMARK

Pre-grasp: 100 / 100
Grasp: 94 / 100
Full pipeline: 94 / 100

The reinforcement-learning policy achieved reliable pre-grasp positioning, but six manipulation attempts still failed during the grasp stage.

GRASP FAILURE ANALYSIS

The six failed grasp attempts were automatically analyzed.

The failures were tightly concentrated around:

X ≈ 0.408 - 0.423 m
Y ≈ -0.111 - -0.097 m

The most important difference between successful and failed grasps was Cartesian descent tracking error.

Successful grasps:

Mean descent error ≈ 0.0057 m

Failed grasps:

Mean descent error ≈ 0.0218 m

The descent error in the failed cases was therefore approximately four times larger.

The analysis indicated that the main problem was not the SAC policy and was also not primarily object alignment.

Instead, the failure was caused by the robot's direct low-X approach trajectory.

ADAPTIVE GRASP RECOVERY

A staged grasp approach was introduced for the difficult low-X region.

Instead of:

Pre-grasp
↓
Direct vertical descent above object
↓
Close gripper

The adaptive planner performs:

Pre-grasp
↓
Move to safe X
↓
Vertical descent
↓
Horizontal inward motion
↓
Close gripper
↓
Lift

This changes the geometry of the robot's approach to the object in the difficult workspace region.

SAFE APPROACH SWEEP

safe_x = 0.440 m -> 0 / 6
safe_x = 0.450 m -> 6 / 6
safe_x = 0.460 m -> 6 / 6
safe_x = 0.470 m -> 6 / 6
safe_x = 0.480 m -> 6 / 6

The selected adaptive approach uses:

safe_x = 0.450 m

This value was the smallest tested safe-X location that successfully recovered all six previously failed grasps.

FINAL ADAPTIVE MANIPULATION BENCHMARK

Evaluation:

100 randomized object placements

Results:

Pre-grasp success: 100 / 100
Grasp success: 100 / 100
Full pipeline: 100 / 100

Grasp strategies:

Direct grasp trials: 89
Direct success: 100%

Staged grasp trials: 11
Staged success: 100%

Improvement over initial manipulation pipeline:

94% -> 100%

+6 percentage points

The adaptive trajectory system therefore recovered the remaining grasp failures after the SAC reaching failures had already been corrected through targeted policy specialization.

FINAL ARCHITECTURE

Randomized Object Position
|
v
Failure-Aware Reach Router
/
Original SAC Targeted SAC
\ /
Pre-Grasp
|
v
Grasp Risk Classifier
/
Normal Region Risky Region
| |
Direct Grasp Staged Approach
|
safe_x = 0.450
|
Vertical Descent
|
Horizontal Inward
\ /
Close Gripper
|
Lift
|
Verify Success

The architecture contains two different failure-aware mechanisms.

Learned-policy failure recovery

General SAC
↓
Failure detection
↓
Targeted SAC specialist
↓
Policy routing

Manipulation trajectory recovery

Grasp failure detection
↓
Risk-region identification
↓
Staged Cartesian trajectory
↓
Successful grasp

KEY TECHNICAL COMPONENTS

PyTorch Soft Actor-Critic implementation
Gaussian stochastic policy
Twin Q-networks
Target Q-networks
Automatic entropy tuning
Experience replay
Polyak target updates
Deterministic policy evaluation
MuJoCo physics simulation
Franka Panda Cartesian control
Jacobian-based Cartesian impedance control
Randomized object placement
Object-relative policy observations
Failure-region mining
Targeted curriculum retraining
General + specialist policy routing
Quantitative rollout analysis
Grasp/lift verification
Adaptive Cartesian trajectory planning
Failure-focused benchmarking

SOFT ACTOR-CRITIC COMPONENTS

GAUSSIAN STOCHASTIC POLICY

The actor learns a Gaussian action distribution.

State
↓
Neural Network
↓
Mean mu(s)
Standard deviation sigma(s)
↓
Sample action
↓
Cartesian dx, dy, dz

The policy learns a probability distribution over Cartesian actions rather than directly producing only a deterministic motion.

TWIN Q-NETWORKS

Two critic networks are used:

Q1(s,a)
Q2(s,a)

Using twin critics reduces the risk of overestimating action values.

The smaller critic estimate can be used when computing the target value:

min(Q1, Q2)

TARGET Q-NETWORKS

Slowly updated copies of the critic networks are maintained.

Online critics
↓
Polyak update
↓
Target critics

The target networks make learning more stable.

AUTOMATIC ENTROPY TUNING

SAC balances:

Task reward
+
Policy exploration

through the entropy term.

The entropy coefficient can be automatically adapted during training rather than manually selecting a fixed value.

EXPERIENCE REPLAY

Transitions collected during robot interaction are stored in a replay buffer.

Each transition can contain:

state
action
reward
next_state
done

During training, random minibatches are sampled from the replay buffer.

POLYAK TARGET UPDATES

Target critic parameters are updated slowly.

Conceptually:

target_parameters =
tau * online_parameters
+
(1 - tau) * target_parameters

This prevents the target network from changing too rapidly.

DETERMINISTIC POLICY EVALUATION

Although SAC uses a stochastic policy during training, evaluation can use the deterministic mean action.

This provides more repeatable policy performance during benchmarking.

OBJECT-RELATIVE REINFORCEMENT LEARNING

Instead of learning only from absolute robot positions, the policy receives information describing the relationship between:

Robot end effector
Object
Pre-grasp target

This allows the policy to learn more general reaching behavior across different randomized object positions.

The policy learns approximately:

"How should I move relative to the object?"

instead of:

"Move to one memorized absolute coordinate."

FAILURE-REGION MINING

After evaluating the policy on randomized unseen targets, failed examples are collected.

Example:

Target 1 -> success
Target 2 -> success
Target 3 -> failure
Target 4 -> failure
Target 5 -> success

The failed target coordinates can then be analyzed spatially.

In this project, failure analysis identified a systematic weakness around the negative-Y region.

This transformed policy improvement from:

Train more everywhere

into:

Train specifically where the policy fails

TARGETED CURRICULUM RETRAINING

Instead of retraining another policy using the original uniform workspace distribution, the specialist policy is trained using samples concentrated around the discovered difficult region.

General workspace training
↓
Policy evaluation
↓
Failure distribution
↓
Targeted sampling
↓
Specialist SAC

This creates a failure-focused curriculum.

GENERAL + SPECIALIST POLICY ROUTING

The system does not discard the original SAC policy.

Instead, both policies are retained:

General SAC
Specialist SAC

A routing rule determines which policy should be used for each object position.

This makes the architecture similar to a small mixture-of-experts system, where different controllers specialize in different parts of the task distribution.

GRASP VERIFICATION

The system does not assume that closing the gripper means that the object was successfully grasped.

Instead, the object's behavior after the grasp is checked.

Close gripper
↓
Lift robot
↓
Observe object position
↓
Did object move upward?
↓
Yes -> grasp success
No -> grasp failure

This provides explicit manipulation verification.

ADAPTIVE CARTESIAN TRAJECTORY PLANNING

For normal workspace regions:

Pre-grasp
↓
Vertical descent
↓
Grasp

For the risky low-X region:

Pre-grasp
↓
Move outward to safe X
↓
Descend vertically
↓
Move horizontally toward object
↓
Grasp

This demonstrates that some robot-learning failures do not require another neural network.

Failure analysis can show that the problem is caused by:

robot kinematics
tracking error
geometry
contact conditions
trajectory design

The system can then apply the appropriate recovery mechanism.

REPOSITORY STRUCTURE

vision-language-panda-pick-place/
|
|-- pickandplace.py
|-- nl_interface.py
|-- world.xml
|-- panda.xml
|
|-- rl/
| |-- panda_rl_env.py
| |-- panda_rl_env_targeted.py
| |
| |-- panda_object_reach_env.py
| |-- panda_object_reach_env_targeted.py
| |
| |-- replay_buffer.py
| |-- sac_networks.py
| |-- sac_agent.py
| |
| |-- train_sac.py
| |-- train_sac_targeted.py
| |-- train_sac_object_reach.py
| |-- train_sac_object_reach_targeted.py
| |
| |-- evaluate_sac.py
| |-- evaluate_sac_targeted.py
| |-- evaluate_sac_hard_targets.py
| |
| |-- evaluate_sac_object_reach.py
| |-- evaluate_sac_object_targeted.py
| |-- evaluate_sac_object_hybrid.py
| |
| |-- analyze_reach_failures.py
| |-- analyze_object_reach_failures.py
| |-- analyze_grasp_failures.py
| |
| |-- hybrid_sac_grasp.py
| |-- evaluate_hybrid_sac_grasp_100.py
| |-- evaluate_hybrid_sac_grasp_adaptive_100.py
| |
| |-- sweep_grasp_corrections.py
| |-- sweep_staged_grasp_approach.py
|
|-- models/
| |-- sac_reach/
| |-- sac_reach_targeted/
| |-- sac_object_reach/
| |-- sac_object_reach_targeted/
| |-- sac_hard_benchmark/
| |-- hybrid_sac_grasp_100/
| |-- hybrid_sac_grasp_adaptive_100/
|
|-- action_encoding/
|-- action_sequence/
|-- perception/
|-- dataset/
|-- diffusion_policy/
|-- world_model/
|-- preference_learning/
|-- video_action/
|
|-- requirements.txt
|-- README.md

RUNNING THE FINAL BENCHMARK

Activate environment:

conda activate panda_rl

Go to project directory:

cd ~/Desktop/vision-language-panda-pick-place

Run:

python rl/evaluate_hybrid_sac_grasp_adaptive_100.py

Expected result:

Pre-grasp success: 100/100
Grasp success: 100/100
Full pipeline success: 100/100

FAILURE-AWARE ROBOT LEARNING CONCEPT

The main idea of this project is that robot learning should not stop after training a single policy.

Policy
↓
Evaluation
↓
Failure discovery
↓
Specialization
↓
Routing
↓
Execution
↓
Verification
↓
Adaptive recovery

This provides a practical framework for improving robot-policy reliability in previously unseen or difficult workspace regions.

Instead of asking only:

"How accurate is my robot policy?"

the system asks:

Where does the policy fail?
Why does it fail?
Can the failure region be modeled?
Can a specialist policy solve it?
Should a different policy be selected there?
Did the manipulation actually succeed?
If not, can the trajectory itself be changed?

This produces a more systematic:

train -> evaluate -> diagnose -> improve -> verify

robot-learning workflow.

TECHNOLOGIES

Python
PyTorch
MuJoCo
Soft Actor-Critic
NumPy
Pandas
Matplotlib
Franka Panda
Cartesian impedance control
Reinforcement learning
Robot manipulation
Failure-aware policy evaluation

EARLIER VLA COMPONENTS

The repository also contains earlier experimental components for:

Natural-language command parsing
Object grounding
Symbolic action encoding
Behavior cloning
Nearest-neighbor imitation
Diffusion-style action prediction
World-model prediction
Preference learning
Video-action datasets

These components form the broader VLA / embodied-AI experimentation layer around the manipulation system.

SHORT TECHNICAL SUMMARY

This project develops a failure-aware reinforcement-learning and manipulation system for the Franka Panda robot in MuJoCo using PyTorch.

A Soft Actor-Critic policy first learns object-relative Cartesian pre-grasp reaching.

Initial evaluation:

88/100 success on unseen targets.

Failure analysis identified a systematic weakness in a negative-Y workspace region.

A targeted SAC specialist trained using failure-focused sampling improved performance in this difficult region from:

22/100 -> 100/100

A failure-aware routing mechanism combining the general and specialist policies then achieved:

100/100 pre-grasp success

on randomized unseen object positions.

When integrated into the complete grasp pipeline, performance initially reached:

94/100

because six grasps failed despite successful pre-grasp positioning.

Automated trajectory analysis showed that failed grasps had significantly larger Cartesian descent errors:

Successful:
approximately 0.0057 m

Failed:
approximately 0.0218 m

The failures were concentrated in a low-X workspace region.

A staged adaptive grasp trajectory was therefore introduced:

pre-grasp
-> safe-X motion
-> vertical descent
-> horizontal inward motion
-> close gripper
-> lift
-> verify

Using:

safe_x = 0.450 m

the final evaluation achieved:

Pre-grasp: 100/100
Grasp: 100/100
Full pipeline: 100/100

The resulting architecture demonstrates a failure-aware robot-learning framework combining reinforcement learning, targeted retraining, specialist-policy routing, quantitative failure analysis, grasp verification, and adaptive trajectory recovery.
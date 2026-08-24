# -*- coding: utf-8 -*-

"""
hybrid_sac_grasp.py

Stage 3 manipulation pipeline

    Real MuJoCo object position
              |
              v
      Failure-aware router
        /             \
   targeted SAC     original SAC
        \             /
         pre-grasp reach
              |
              v
      scripted precision align
              |
              v
         scripted descend
              |
              v
          close gripper
              |
              v
             lift
              |
              v
        grasp verification


IMPORTANT:
    The SAC policy is responsible for reaching the object-relative
    pre-grasp region.

    The final grasp is intentionally scripted for this stage.

    Later stages can replace transport/place components with learned
    policies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import torch


# ================================================================
# Project imports
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl.panda_object_reach_env import PandaObjectReachEnv
from rl.sac_agent import SACAgent, SACConfig


# ================================================================
# Configuration
# ================================================================

ROUTER_Y_THRESHOLD = -0.10

MAX_RL_STEPS = 120

# ------------------------------------------------
# Precision grasp parameters
# ------------------------------------------------

# Panda hand body position is above the fingertips.
# We begin conservatively here and can tune after the first test.
GRASP_HAND_OFFSET_Z = 0.115

# Lift hand by this amount after closing.
LIFT_DISTANCE = 0.18

# Number of interpolation points for scripted movements.
ALIGN_SEGMENTS = 30
DESCEND_SEGMENTS = 45
LIFT_SEGMENTS = 45

# Physics integration steps for every interpolation target.
PHYSICS_STEPS_PER_SEGMENT = 20

# Let gripper settle after close.
GRIPPER_SETTLE_STEPS = 250

# Verification criterion:
# object must move at least this much vertically.
MIN_OBJECT_LIFT = 0.05


ORIGINAL_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach"
    / "sac_object_reach_best.pt"
)

TARGETED_MODEL = (
    PROJECT_ROOT
    / "models"
    / "sac_object_reach_targeted"
    / "sac_object_targeted_best.pt"
)


# ================================================================
# SAC loader
# ================================================================

def load_agent(
    model_path: Path,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> SACAgent:

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    config = SACConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        initial_alpha=0.2,
    )

    agent = SACAgent(
        config=config,
        device=device,
    )

    agent.load(
        model_path
    )

    agent.actor.eval()
    agent.critics.eval()
    agent.target_critics.eval()

    return agent


# ================================================================
# Physics helper
# ================================================================

def physics_step(
    env: PandaObjectReachEnv,
    steps: int,
) -> None:

    for _ in range(steps):

        env.robot.control(
            env.robot.target_pos,
            env.robot.target_quat,
        )

        mujoco.mj_step(
            env.robot.model,
            env.robot.data,
        )


# ================================================================
# Smooth Cartesian scripted movement
# ================================================================

def move_hand_smooth(
    env: PandaObjectReachEnv,
    target_position: np.ndarray,
    segments: int,
    physics_steps_per_segment: int,
    label: str,
) -> None:

    start = env._ee_position().astype(float)

    target = np.asarray(
        target_position,
        dtype=float,
    )

    print(
        f"\n[{label}]"
    )

    print(
        "Start:",
        np.round(start, 4)
    )

    print(
        "Target:",
        np.round(target, 4)
    )

    for i in range(
        1,
        segments + 1,
    ):

        alpha = (
            i / segments
        )

        desired = (
            start
            + alpha * (
                target - start
            )
        )

        env.robot.target_pos = (
            desired.copy()
        )

        # Maintain original hand orientation.
        env.robot.target_quat = (
            env.robot.home_quat.copy()
        )

        physics_step(
            env,
            physics_steps_per_segment,
        )

    final = env._ee_position()

    error = float(
        np.linalg.norm(
            target - final
        )
    )

    print(
        "Reached:",
        np.round(final, 4)
    )

    print(
        f"Position error: "
        f"{error:.4f} m"
    )


# ================================================================
# Hybrid SAC pre-grasp
# ================================================================

def run_hybrid_pregrasp(
    env: PandaObjectReachEnv,
    original_agent: SACAgent,
    targeted_agent: SACAgent,
):

    state, info = env.reset()

    obj = np.asarray(
        info["object_position"],
        dtype=np.float32,
    )

    goal = np.asarray(
        info["goal_position"],
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # Failure-aware routing
    # ------------------------------------------------------------

    if obj[1] < ROUTER_Y_THRESHOLD:

        agent = targeted_agent
        policy_name = "TARGETED SAC"

    else:

        agent = original_agent
        policy_name = "ORIGINAL SAC"

    print(
        "\n"
        "============================================================"
    )

    print(
        "HYBRID SAC PRE-GRASP"
    )

    print(
        "============================================================"
    )

    print(
        "Object position:",
        np.round(obj, 4)
    )

    print(
        "Pre-grasp goal:",
        np.round(goal, 4)
    )

    print(
        "Selected policy:",
        policy_name
    )

    print(
        "Initial EE:",
        np.round(
            info["ee_position"],
            4,
        )
    )

    print(
        f"Initial distance: "
        f"{info['distance']:.4f} m"
    )

    success = False
    final_info = info

    for step in range(
        1,
        MAX_RL_STEPS + 1,
    ):

        action = agent.select_action(
            state,
            deterministic=True,
        )

        (
            state,
            reward,
            terminated,
            truncated,
            final_info,
        ) = env.step(
            action
        )

        if (
            step <= 5
            or step % 10 == 0
            or terminated
        ):

            print(
                f"RL step={step:03d} | "
                f"distance="
                f"{final_info['distance']:.4f} | "
                f"reward="
                f"{reward:+.3f}"
            )

        if terminated:

            success = True
            break

        if truncated:
            break

    print(
        "\nPre-grasp policy:",
        policy_name
    )

    print(
        "Pre-grasp success:",
        success
    )

    print(
        "Final EE:",
        np.round(
            final_info["ee_position"],
            4,
        )
    )

    print(
        f"Final distance: "
        f"{final_info['distance']:.4f} m"
    )

    return (
        success,
        policy_name,
        final_info,
    )


# ================================================================
# Scripted grasp
# ================================================================

def execute_scripted_grasp(
    env: PandaObjectReachEnv,
):

    print(
        "\n"
        "============================================================"
    )

    print(
        "SCRIPTED GRASP"
    )

    print(
        "============================================================"
    )

    obj_before = (
        env._object_position()
        .astype(float)
    )

    hand_before = (
        env._ee_position()
        .astype(float)
    )

    print(
        "Object before grasp:",
        np.round(
            obj_before,
            4,
        )
    )

    print(
        "Hand before grasp:",
        np.round(
            hand_before,
            4,
        )
    )

    # ============================================================
    # Phase A:
    # Precision XY alignment while keeping current hand height.
    # ============================================================

    alignment_target = np.array(
        [
            obj_before[0],
            obj_before[1],
            hand_before[2],
        ],
        dtype=float,
    )

    move_hand_smooth(
        env=env,
        target_position=alignment_target,
        segments=ALIGN_SEGMENTS,
        physics_steps_per_segment=PHYSICS_STEPS_PER_SEGMENT,
        label="A - PRECISION XY ALIGNMENT",
    )

    # ============================================================
    # Phase B:
    # Descend vertically.
    #
    # Object center ~0.03 m.
    # Hand body must remain above object/fingers.
    # ============================================================

    obj_now = (
        env._object_position()
        .astype(float)
    )

    grasp_target = np.array(
        [
            obj_now[0],
            obj_now[1],
            obj_now[2]
            + GRASP_HAND_OFFSET_Z,
        ],
        dtype=float,
    )

    move_hand_smooth(
        env=env,
        target_position=grasp_target,
        segments=DESCEND_SEGMENTS,
        physics_steps_per_segment=PHYSICS_STEPS_PER_SEGMENT,
        label="B - VERTICAL DESCENT",
    )

    # ============================================================
    # Phase C:
    # Close gripper
    # ============================================================

    print(
        "\n[C - CLOSE GRIPPER]"
    )

    print(
        "Closing gripper..."
    )

    # In the existing Demo:
    # True = open
    # False = close
    env.robot.gripper(
        False
    )

    physics_step(
        env,
        GRIPPER_SETTLE_STEPS,
    )

    object_after_close = (
        env._object_position()
        .astype(float)
    )

    hand_after_close = (
        env._ee_position()
        .astype(float)
    )

    print(
        "Hand after close:",
        np.round(
            hand_after_close,
            4,
        )
    )

    print(
        "Object after close:",
        np.round(
            object_after_close,
            4,
        )
    )

    # ============================================================
    # Phase D:
    # Lift
    # ============================================================

    print(
        "\n[D - LIFT]"
    )

    current_hand = (
        env._ee_position()
        .astype(float)
    )

    lift_target = (
        current_hand.copy()
    )

    lift_target[2] += (
        LIFT_DISTANCE
    )

    move_hand_smooth(
        env=env,
        target_position=lift_target,
        segments=LIFT_SEGMENTS,
        physics_steps_per_segment=PHYSICS_STEPS_PER_SEGMENT,
        label="D - LIFT OBJECT",
    )

    # Let physics settle.
    physics_step(
        env,
        150,
    )

    # ============================================================
    # Verification
    # ============================================================

    object_after_lift = (
        env._object_position()
        .astype(float)
    )

    hand_after_lift = (
        env._ee_position()
        .astype(float)
    )

    object_vertical_motion = float(
        object_after_lift[2]
        - obj_before[2]
    )

    object_xy_motion = float(
        np.linalg.norm(
            object_after_lift[:2]
            - obj_before[:2]
        )
    )

    hand_object_distance = float(
        np.linalg.norm(
            hand_after_lift
            - object_after_lift
        )
    )

    grasp_success = bool(
        object_vertical_motion
        >= MIN_OBJECT_LIFT
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "GRASP VERIFICATION"
    )

    print(
        "============================================================"
    )

    print(
        "Object initial:",
        np.round(
            obj_before,
            4,
        )
    )

    print(
        "Object final:",
        np.round(
            object_after_lift,
            4,
        )
    )

    print(
        "Hand final:",
        np.round(
            hand_after_lift,
            4,
        )
    )

    print(
        f"\nObject vertical motion: "
        f"{object_vertical_motion:.4f} m"
    )

    print(
        f"Object XY motion: "
        f"{object_xy_motion:.4f} m"
    )

    print(
        f"Final hand-object distance: "
        f"{hand_object_distance:.4f} m"
    )

    print(
        f"Required object lift: "
        f"{MIN_OBJECT_LIFT:.4f} m"
    )

    print(
        "\nGRASP SUCCESS:",
        grasp_success
    )

    return {
        "success":
            grasp_success,

        "object_initial":
            obj_before,

        "object_final":
            object_after_lift,

        "object_vertical_motion":
            object_vertical_motion,

        "hand_object_distance":
            hand_object_distance,
    }


# ================================================================
# Main
# ================================================================

def main():

    print(
        "\n"
        "============================================================"
    )

    print(
        "Hybrid SAC + Scripted Grasp"
    )

    print(
        "============================================================"
    )

    # ============================================================
    # Device
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ============================================================
    # Environment
    # ============================================================

    env = PandaObjectReachEnv(
        object_name="red_box",
        max_steps=MAX_RL_STEPS,
        action_scale=0.025,
        physics_steps_per_action=40,
        success_threshold=0.040,
        pregrasp_height=0.15,
        seed=2028,
    )

    try:

        # ========================================================
        # Load both policies
        # ========================================================

        print(
            "\nLoading original SAC..."
        )

        original_agent = load_agent(
            ORIGINAL_MODEL,
            env.observation_dim,
            env.action_dim,
            device,
        )

        print(
            "Loading targeted SAC..."
        )

        targeted_agent = load_agent(
            TARGETED_MODEL,
            env.observation_dim,
            env.action_dim,
            device,
        )

        # ========================================================
        # Stage 1:
        # Hybrid RL pre-grasp
        # ========================================================

        (
            pregrasp_success,
            policy_name,
            pregrasp_info,
        ) = run_hybrid_pregrasp(
            env,
            original_agent,
            targeted_agent,
        )

        if not pregrasp_success:

            print(
                "\n"
                "Pre-grasp failed."
            )

            print(
                "Grasp sequence aborted."
            )

            return

        # ========================================================
        # Stage 2:
        # Scripted grasp + lift
        # ========================================================

        grasp_result = (
            execute_scripted_grasp(
                env
            )
        )

        # ========================================================
        # Final
        # ========================================================

        print(
            "\n"
            "============================================================"
        )

        print(
            "PIPELINE RESULT"
        )

        print(
            "============================================================"
        )

        print(
            "Selected SAC policy:",
            policy_name
        )

        print(
            "RL pre-grasp:",
            "SUCCESS"
            if pregrasp_success
            else "FAIL"
        )

        print(
            "Scripted grasp:",
            "SUCCESS"
            if grasp_result[
                "success"
            ]
            else "FAIL"
        )

        if (
            pregrasp_success
            and grasp_result["success"]
        ):

            print(
                "\nFULL PIPELINE: SUCCESS"
            )

        else:

            print(
                "\nFULL PIPELINE: FAILED"
            )

        print(
            "============================================================\n"
        )

    finally:

        env.close()


if __name__ == "__main__":
    main()

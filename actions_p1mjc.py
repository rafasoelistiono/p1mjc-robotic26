import time
import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "scene.xml"   # scene.xml 

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data  = mujoco.MjData(model)
DT = model.opt.timestep

# -------------------------
# Names & mapping
JOINT_ACTS = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint",
]

def obj_id(objtype, name):
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise RuntimeError(f"Not found: {name}")
    return int(i)

ACT = {n: obj_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_ACTS}
QADR = {n: int(model.jnt_qposadr[obj_id(mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINT_ACTS}

# -------------------------
# Small utils
def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x*x*(3 - 2*x)

def get_angles():
    return {n: float(data.qpos[QADR[n]]) for n in JOINT_ACTS}

def set_ctrl(angles):
    for n, v in angles.items():
        a = ACT[n]
        lo, hi = model.actuator_ctrlrange[a]
        data.ctrl[a] = float(np.clip(v, lo, hi))

def lerp_pose(a0, a1, s):
    return {n: (1-s)*a0[n] + s*a1.get(n, a0[n]) for n in JOINT_ACTS}

def lock_base(base7):
    data.qpos[:7] = base7
    data.qvel[:6] = 0.0

# -------------------------
# Reset + prone
def reset_to_stand():
    for k in ("stand", "home", "knees_bent"):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, k)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(model, data, kid)
            mujoco.mj_forward(model, data)
            return
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

PRONE_POS  = np.array([0.0, 0.0, 0.05])
PRONE_QUAT = np.array([0.7071, 0.0, -0.7071, 0.0])

def set_prone():
    data.qpos[0:3] = PRONE_POS
    data.qpos[3:7] = PRONE_QUAT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.qpos[:7].copy()

# -------------------------
# Motion
def goto(viewer, target, T, base_ref=None, lock=False):
    start = get_angles()
    steps = max(1, int(T / DT))
    for i in range(steps):
        s = smoothstep((i + 1) / steps)
        set_ctrl(lerp_pose(start, target, s))
        mujoco.mj_step(model, data)
        if lock and base_ref is not None:
            lock_base(base_ref)
            mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(DT)

def hold(viewer, target, T, base_ref=None, lock=False):
    steps = max(1, int(T / DT))
    set_ctrl(target)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if lock and base_ref is not None:
            lock_base(base_ref)
            mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(DT)

# -------------------------
# STEP_ANGLES 
FLIP_PITCH = True
P = -1.0 if FLIP_PITCH else 1.0

ARM_RELAX = {
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.0, "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
}

WAIST_ENABLE = True
WIDE_HIP_ROLL_MID = np.deg2rad(12)
WIDE_HIP_ROLL_MAX = np.deg2rad(40)
ANKLE_ROLL_COMP = 0.45

def waist_pose(pitch_deg=0.0, roll_deg=0.0, yaw_deg=0.0):
    if not WAIST_ENABLE:
        return {"waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0}
    return {
        "waist_yaw_joint":   np.deg2rad(yaw_deg),
        "waist_roll_joint":  np.deg2rad(roll_deg),
        "waist_pitch_joint": np.deg2rad(pitch_deg),
    }

def legs_pose(hip_pitch, knee, ankle_pitch, hip_roll=0.0):
    ankle_roll = float(np.clip(-ANKLE_ROLL_COMP * hip_roll, np.deg2rad(-14), np.deg2rad(14)))
    return {
        "left_hip_pitch_joint":    P * hip_pitch,
        "left_hip_roll_joint":     hip_roll,
        "left_hip_yaw_joint":      0.0,
        "left_knee_joint":         knee,
        "left_ankle_pitch_joint":  P * ankle_pitch,
        "left_ankle_roll_joint":   ankle_roll,
        "right_hip_pitch_joint":   P * hip_pitch,
        "right_hip_roll_joint":    -hip_roll,
        "right_hip_yaw_joint":     0.0,
        "right_knee_joint":        knee,
        "right_ankle_pitch_joint": P * ankle_pitch,
        "right_ankle_roll_joint":  -ankle_roll,
    }

def step(hip_pitch, knee, ankle_pitch, hip_roll=0.0, waist_pitch_deg=0.0):
    return {**ARM_RELAX, **waist_pose(waist_pitch_deg), **legs_pose(hip_pitch, knee, ankle_pitch, hip_roll)}

STEP_ANGLES = [
    step(0.2,             0.2,             -0.1,              0.0, 0),
    step(np.deg2rad(60),  np.deg2rad(90),   0.0,               0.0, 0),
    step(np.deg2rad(120), np.deg2rad(120),  np.deg2rad(-10),   0.0, 0),
    step(np.deg2rad(110), np.deg2rad(0),    np.deg2rad(10),    0.0, 0),
    step(np.deg2rad(120), np.deg2rad(120),  0.0,               0.0, 0),

    step(np.deg2rad(-10),  np.deg2rad(90),  np.deg2rad(-5),    np.deg2rad(8),  8),
    step(np.deg2rad(-70),  np.deg2rad(90),  np.deg2rad(20),    WIDE_HIP_ROLL_MID, 14),
    step(np.deg2rad(-80),  np.deg2rad(90),  np.deg2rad(20),    WIDE_HIP_ROLL_MAX, 14),
    step(np.deg2rad(-90),  np.deg2rad(100), np.deg2rad(40),    WIDE_HIP_ROLL_MAX, 10),
    step(np.deg2rad(-100), np.deg2rad(100), np.deg2rad(40),    WIDE_HIP_ROLL_MAX, 20),
    step(np.deg2rad(-100), np.deg2rad(100), np.deg2rad(40),    WIDE_HIP_ROLL_MAX, 22),
    step(np.deg2rad(-90),  np.deg2rad(110), np.deg2rad(45),    WIDE_HIP_ROLL_MAX, 25),
]

TIMINGS = [0.70, 0.28, 0.28, 0.28, 0.28, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]
PRE_HOLD, HOLD_BETWEEN = 0.15, 0.10

if len(TIMINGS) != len(STEP_ANGLES):
    raise RuntimeError("TIMINGS harus sama panjang dengan STEP_ANGLES")

# -------------------------
# Run
# -------------------------
reset_to_stand()
pending = False

def key_cb(k):
    global pending
    if k == ord('1'):
        pending = True


with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
    while viewer.is_running():
        if pending:
            pending = False

            base_ref = set_prone()

            # ctrl awal
            set_ctrl(STEP_ANGLES[0])
            hold(viewer, STEP_ANGLES[0], PRE_HOLD, base_ref, lock=True)

            # step sequence
            lock_per_step = [True]*6 + [False]*(len(STEP_ANGLES)-6)
            for i, tgt in enumerate(STEP_ANGLES):
                lock = lock_per_step[i]
                goto(viewer, tgt, TIMINGS[i], base_ref, lock=lock)
                hold(viewer, tgt, HOLD_BETWEEN, base_ref, lock=lock)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(DT)

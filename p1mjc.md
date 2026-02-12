# P1MJC Notes (MuJoCo Unitree G1)

## 1. Konsep Fondasi MuJoCo

MuJoCo punya dua objek utama:

### `MjModel`
Menyimpan struktur dan konfigurasi robot:
- daftar `joint`, `actuator`, `sensor`
- parameter fisika (massa, inersia, friction, timestep, dll.)
- `ctrlrange` tiap actuator
- `keyframe` 

### `MjData`
Menyimpan state simulasi yang sedang berjalan:
- `data.qpos`: posisi semua DOF (termasuk floating base)
- `data.qvel`: velocity semua DOF
- `data.ctrl`: input ke actuator 
- `data.time`: waktu simulasi

## 2. Istilah Penting: Joint, Actuator, Range, Keyframe

### A. Joint
Joint adalah derajat kebebasan (DOF) yang bisa bergerak.

Contoh di G1:
- `left_knee_joint`
- `waist_pitch_joint`

Di MuJoCo:
- nama joint bisa dicek lewat `mj_id2name(model, mjOBJ_JOINT, ...)`
- alamat joint di `qpos` bisa dicek lewat `model.jnt_qposadr[jid]`

### B. Actuator
Actuator adalah motor/controller yang memberi perintah ke joint.

Contoh:
- actuator `left_knee_joint` menggerakkan joint `left_knee_joint`

Di MuJoCo:
- input actuator masuk lewat `data.ctrl[act_id]`
- hubungan actuator ke joint biasanya ada di `model.actuator_trnid`

Catatan:
- tidak semua joint harus punya actuator (bisa ada joint pasif)

### C. Ctrl Range
`ctrlrange` adalah batas nilai kontrol yang boleh diberikan ke actuator.

Contoh:
- `waist_pitch_joint` hanya boleh `[-0.52, 0.52]` rad

Jika target di luar range:
- kontrol akan di-clamp
- gerakan tidak sesuai rencana

### D. Keyframe
Keyframe adalah pose referensi yang disimpan di XML.

Contoh:
- `home`
- `knees_bent`

Kegunaan keyframe:
- reset ke pose awal yang stabil
- jadi baseline untuk gerakan berikutnya

Reset ke keyframe:

```python
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)
```

## 3. Inspect Model (`inspect_g1.py`)

Script ini dipakai untuk melihat:
- daftar joint
- daftar actuator dan joint yang digerakkan
- `ctrlrange` tiap actuator

Contoh run:

```bash
python inspect_g1.py --xml scene.xml
python inspect_g1.py --xml scene_mjx.xml
python inspect_g1.py --xml scene_with_hands.xml
```

## 4. Contoh Gerakan Sederhana (`one_move.py`)

Alur gerakan:
- set ke posisi tidur
- tekuk lutut kiri dan kanan sedikit

Jalankan:

```bash
python one_move.py
```

## Lampiran A: `inspect_g1.py` (lengkap)

```python
import mujoco
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "scene.xml"  # ganti kalau pakai scene_with_hands.xml

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

print("=== JOINTS ===")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = model.jnt_type[i]
    print(i, name, "type", int(jtype), "qposadr", int(model.jnt_qposadr[i]), "dofadr", int(model.jnt_dofadr[i]))

print("\n=== ACTUATORS ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    trn_jnt = int(model.actuator_trnid[i, 0])
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trn_jnt) if trn_jnt >= 0 else None
    print(i, name, "-> joint:", jname)
```

## Lampiran B: `one_move.py` (contoh)

```python
import time
import numpy as np
from pathlib import Path
import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
DT = model.opt.timestep

JOINT_ACTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

def _id(objtype, name):
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise RuntimeError(f"Not found: {name}")
    return int(i)

ACT = {n: _id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_ACTS}
QADR = {n: int(model.jnt_qposadr[_id(mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINT_ACTS}

def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)

def get_angles():
    return {n: float(data.qpos[QADR[n]]) for n in JOINT_ACTS}

def set_ctrl(pose):
    for n, v in pose.items():
        a = ACT[n]
        lo, hi = model.actuator_ctrlrange[a]
        data.ctrl[a] = float(np.clip(v, lo, hi))

def lerp_pose(a0, a1, s):
    return {n: (1 - s) * a0[n] + s * a1.get(n, a0[n]) for n in JOINT_ACTS}

def goto(viewer, target, T):
    start = get_angles()
    steps = max(1, int(T / DT))
    for i in range(steps):
        s = smoothstep((i + 1) / steps)
        set_ctrl(lerp_pose(start, target, s))
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(DT)

def hold(viewer, target, T):
    steps = max(1, int(T / DT))
    set_ctrl(target)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(DT)

PRONE_POS = np.array([0.0, 0.0, 0.05], dtype=float)
PRONE_QUAT = np.array([0.7071, 0.0, -0.7071, 0.0], dtype=float)

def set_prone():
    data.qpos[0:3] = PRONE_POS
    data.qpos[3:7] = PRONE_QUAT
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

def legs_pose(
    l_hip_pitch=0.0, l_hip_roll=0.0, l_hip_yaw=0.0, l_knee=0.0, l_ankle_pitch=0.0, l_ankle_roll=0.0,
    r_hip_pitch=0.0, r_hip_roll=0.0, r_hip_yaw=0.0, r_knee=0.0, r_ankle_pitch=0.0, r_ankle_roll=0.0,
):
    return {
        "left_hip_pitch_joint": l_hip_pitch,
        "left_hip_roll_joint": l_hip_roll,
        "left_hip_yaw_joint": l_hip_yaw,
        "left_knee_joint": l_knee,
        "left_ankle_pitch_joint": l_ankle_pitch,
        "left_ankle_roll_joint": l_ankle_roll,
        "right_hip_pitch_joint": r_hip_pitch,
        "right_hip_roll_joint": r_hip_roll,
        "right_hip_yaw_joint": r_hip_yaw,
        "right_knee_joint": r_knee,
        "right_ankle_pitch_joint": r_ankle_pitch,
        "right_ankle_roll_joint": r_ankle_roll,
    }

RELAX = legs_pose()
RIGHT_UP = legs_pose(
    r_hip_pitch=np.deg2rad(-35),
    r_knee=np.deg2rad(55),
    r_ankle_pitch=np.deg2rad(10),
)
LEFT_UP = legs_pose(
    l_hip_pitch=np.deg2rad(-35),
    l_knee=np.deg2rad(55),
    l_ankle_pitch=np.deg2rad(10),
)

pending = False

def key_cb(k):
    global pending
    if k == ord("1"):
        pending = True

with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
    while viewer.is_running():
        if pending:
            pending = False

            set_prone()
            set_ctrl(RELAX)
            hold(viewer, RELAX, 0.2)

            goto(viewer, RIGHT_UP, 0.7)
            hold(viewer, RIGHT_UP, 0.3)
            goto(viewer, RELAX, 0.5)
            hold(viewer, RELAX, 0.1)

            goto(viewer, LEFT_UP, 0.7)
            hold(viewer, LEFT_UP, 0.3)
            goto(viewer, RELAX, 0.5)
            hold(viewer, RELAX, 0.1)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(DT)
```

## Target Progress

Wajib: `0.03-0.10`

Output minimal yang diharapkan:
- bisa menjalankan MuJoCo viewer untuk `scene.xml` / `scene_mjx.xml`
- bisa inspect joint, actuator, range, keyframe
- robot dalam posisi "tiduran" (`set_prone`)
- robot melakukan gerakan sesuai video `0.03-0.10`

Bonus (opsional): `0.11-0.15`

## Rubrik Penilaian (Matriks Capaian)

| Komponen | Deskripsi | Bobot |
|---|---|---|
| 1 | Mampu mengetahui joint dan actuator yang diperlukan | 25% |
| 2 | Mampu manipulasi gerakan dan memperhitungkan sudut gerakan (radian) | 25% |
| 3 | Mampu membuat gerakan mulus dengan fungsi `smooth` | 25% |
| 4 | Membuat dokumentasi yang baik | 25% |
| Bonus | Implementasi `0.11-0.15` | +5% |

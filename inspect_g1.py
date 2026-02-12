import mujoco
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "scene.xml"   # ganti kalau pakai scene_with_hands.xml

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

import time
import mujoco
import mujoco.viewer

MODEL_PATH = "scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
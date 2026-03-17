import time
import numpy as np
import mujoco
import mujoco.viewer

from mujoco_mpc import agent as agent_lib

XML_PATH = "/Users/dc/mujoco_mpc/mjpc/tasks/quadruped/task_flat.xml"


def push_state_to_agent(agent, model, data):
    agent.set_state(
        time=float(data.time),
        qpos=data.qpos.tolist(),
        qvel=data.qvel.tolist(),
        act=data.act.tolist() if model.na > 0 else [],
        mocap_pos=data.mocap_pos.reshape(-1).tolist() if model.nmocap > 0 else [],
        mocap_quat=data.mocap_quat.reshape(-1).tolist() if model.nmocap > 0 else [],
        userdata=data.userdata.tolist() if model.nuserdata > 0 else [],
    )


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)

    with agent_lib.Agent(task_id="Quadruped Flat", model=model) as agent:
        push_state_to_agent(agent, model, data)

        params = agent.get_task_parameters()
        print("Initial task parameters:")
        print(params)

        # Adjust behavior here
        new_params = dict(params)
        if "Walk speed" in new_params:
            new_params["Walk speed"] = 1.0
        if "Walk turn" in new_params:
            new_params["Walk turn"] = 0.2
        agent.set_task_parameters(new_params)

        weights = agent.get_cost_weights()
        print("\nInitial cost weights:")
        print(weights)

        if "Position" in weights:
            new_weights = dict(weights)
            new_weights["Position"] = float(weights["Position"]) * 1.2
            agent.set_cost_weights(new_weights)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.sync()

            while viewer.is_running():
                # Replan from current state
                agent.planner_step()

                # Apply first action of optimized plan
                action = np.array(agent.get_action(), dtype=np.float64)
                if model.nu > 0:
                    data.ctrl[:] = action

                mujoco.mj_step(model, data)

                # Send new sim state back to planner
                push_state_to_agent(agent, model, data)

                # Optional camera tweak
                viewer.sync()

                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()


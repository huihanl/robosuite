import numpy as np
import sys

data_path = sys.argv[1]
data = np.load(data_path, allow_pickle=True)

new_data = []

for traj in data:
    for t in range(len(traj["observations"])):
        o = traj["observations"][t]
        state_info = np.concatenate([o["robot0_eef_pos"], 
                                     o["robot0_eef_quat"], 
                                     o["robot0_gripper_qpos"], 
                                     o["object-state"]])
        print(state_info)
        traj["observations"][t] = {"state": state_info}

        next_o = traj["next_observations"][t]
        next_state_info = np.concatenate([next_o["robot0_eef_pos"],
                                          next_o["robot0_eef_quat"],
                                          next_o["robot0_gripper_qpos"],
                                          next_o["object-state"]])
        traj["next_observations"][t] = {"state": next_state_info}
        
        new_data.append(traj)

np.save("{}_cleaned".format(data_path[:-4]), new_data)

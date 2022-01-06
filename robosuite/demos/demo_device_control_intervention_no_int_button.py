"""Teleoperate robot with keyboard or SpaceMouse. """

import argparse
import numpy as np
import os
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper, DataCollectionWrapper
from tqdm import tqdm
import time
import numpy as np
from datetime import datetime
import torch
import csv
import json
import h5py
import shutil
from glob import glob

GOOD_EPISODE_LENGTH = 250 - 1
MAX_EPISODE_LENGTH = 400 - 1
SUCCESS_HOLD = 5

class RandomPolicy():
    def __init__(self, env):
        self.env = env
        self.low, self.high = env.action_spec

    def get_action(self, obs):
        return np.random.uniform(self.low, self.high) / 2

class TrainedPolicy():
    def __init__(self, checkpoint):
        params = torch.load(checkpoint, map_location='cpu')
        self.policy = params['evaluation/policy']
    def get_action(self, obs):
        return self.policy.get_action(obs)[0]

def is_empty_input_spacemouse(action):
    empty_input1 = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -1.000])
    empty_input2 = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000])
    if np.array_equal(np.abs(action), empty_input2) or np.array_equal(np.abs(action), empty_input2):
        return True
    return False

def post_process_state(obs):
    state_info = np.concatenate([obs["robot0_eef_pos"],
                                 obs["robot0_eef_quat"],
                                 obs["robot0_gripper_qpos"],
                                 obs["object-state"]])
    final_obs = {"state": state_info}
    return final_obs

def write_to_csv(csv_path, data):
    # open the file in the write mode
    with open(csv_path, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def write_header(csv_path, header):
    write_to_csv(csv_path, header)

def terminate_condition_met(time_success, timestep_count, term_cond):
    assert term_cond in ["fixed_length", "success_count"]
    if term_cond == "fixed_length":
        return timestep_count >= GOOD_EPISODE_LENGTH and time_success > 0
    elif term_cond == "success_count":
        return time_success == SUCCESS_HOLD

def fix_spacemouse_action(action, grasp, last_grasp):
    """ Fixing Spacemouse Action """
    # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
    # toggle arm control and / or camera viewing angle if requested
    if last_grasp < 0 < grasp:
        if args.switch_on_grasp:
            args.arm = "left" if args.arm == "right" else "right"
        if args.toggle_camera_on_grasp:
            cam_id = (cam_id + 1) % num_cam
            env.viewer.set_camera(camera_id=cam_id)
    # Update last grasp
    last_grasp = grasp

    # Fill out the rest of the action space if necessary
    rem_action_dim = env.action_dim - action.size
    if rem_action_dim > 0:
        # Initialize remaining action space
        rem_action = np.zeros(rem_action_dim)
        # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
        if args.arm == "right":
            action = np.concatenate([action, rem_action])
        elif args.arm == "left":
            action = np.concatenate([rem_action, action])
        else:
            # Only right and left arms supported
            print("Error: Unsupported arm specified -- "
                  "must be either 'right' or 'left'! Got: {}".format(args.arm))
    elif rem_action_dim < 0:
        # We're in an environment with no gripper action space, so trim the action space to be the action dim
        action = action[:env.action_dim]

    """ End Fixing Spacemouse Action """
    return action, last_grasp

def gather_demonstrations_as_hdf5(directory, out_dir, env_info, remove_directory=[]):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        print(ep_directory)
        if ep_directory in remove_directory:
            print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

def collect_trajectory(env, device, args, data, remove_directory):

    if len(data) % 5 == 0:

        save_name = "/home/huihanl/{}-iter_{}-n_{}_{}.npy".format(args.environment,
                                                                  args.training_iter,
                                                                  len(data), datetime.now().strftime("%m_%d_%H_%M_%S"))
        np.save(save_name, data)

    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        dense_rewards=[],
        next_observations=[],
        terminals=[],
        is_human=[],
    )

    # Reset the environment
    obs = env.reset()
    obs = post_process_state(obs)

    #time.sleep(2)

    # Setup rendering
    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    env.render()

    # Initialize variables that should the maintained between resets
    last_grasp = 0

    # Initialize device control
    device.start_control()

    time_success = 0

    timestep_count = 0

    this_human_sample = 0

    first_nonzero = False

    saving = True

    success_at_time = -1

    grasped = False

    while True:
        if_sleep = False if time_success > 0 else True

        if if_sleep:
            time.sleep(0.05)

        # Set active robot
        active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=args.arm,
            env_configuration=args.config
        )

        if action is None:
            saving = False
            break

        action, last_grasp = fix_spacemouse_action(action, grasp, last_grasp)

        if is_empty_input_spacemouse(action):
            if args.all_demos:
                if not first_nonzero: # if have not seen nonzero action, should not be zero action
                    continue # if all demos, no action
                # else: okay to be zero action afterwards
                this_human_sample += 1
            else:
                action = policy.get_action(obs["state"]) # if not all demos, use agent action
            is_human = 0
        else:
            first_nonzero = True
            this_human_sample += 1
            if args.all_demos:
                is_human = 0 # iter 0 is viewed as non-intervention
            else:
                is_human = 1
        
        if grasped and action[-1] < -0.7:
            grasped = False
            grasp_timesteps = 0
            while True:
                grasp_timesteps += 1
                action_human, grasp = input2action(
                    device=device,
                    robot=active_robot,
                    active_arm=args.arm,
                    env_configuration=args.config
                )
                if grasp_timesteps > 10 or not grasp:
                    break
            if grasp:
                continue

        elif not grasped and action[-1] > 0.7:
            grasped = True

        # Step through the simulation and render
        next_obs, reward, done, info = env.step(action)

        next_obs = post_process_state(next_obs)

        traj["observations"].append(obs)
        traj["actions"].append(action)
        traj["rewards"].append(0.0 if reward < 1.0 else 1.0)
        traj["dense_rewards"].append(reward)
        traj["next_observations"].append(next_obs)
        traj["terminals"].append(done)
        traj["is_human"].append(is_human)
        obs = next_obs

        env.render()

        if env._check_success():
            time_success += 1
            if time_success == 1:
                print("Success length: ", timestep_count)
                success_at_time = timestep_count

        if terminate_condition_met(time_success=time_success,
                                   timestep_count=timestep_count,
                                   term_cond=args.term_condition):
            data.append(traj)
            break

        timestep_count += 1
        if timestep_count > MAX_EPISODE_LENGTH:
            print("discard this trial")
            saving = False
            break

    if not saving:
        remove_directory.append(env.ep_directory.split('/')[-1])
    env.close()
    return saving, this_human_sample, success_at_time, traj

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="NutAssembly")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--pos-sensitivity", type=float, default=1.5, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.8, help="How much to scale rotation user inputs")
    parser.add_argument("--num-trajectories", type=int, default=20, help="Number of trajectories to collect / evaluate")
    parser.add_argument("--training-iter", type=int)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--csv-filename", type=str, default="")
    parser.add_argument("--all-demos", action="store_true", default=False)
    parser.add_argument("--term-condition", default="fixed_length", type=str)
    parser.add_argument("--base-dir", default="/home/huihanl/HITL_data", type=str)
    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == 'ik':
        controller_name = 'IK_POSE'
    elif args.controller == 'osc':
        controller_name = 'OSC_POSE'
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        # "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # if args.environment == "NutAssembly":
    #     config["single_object_mode"] = 2
    #     config["nut_type"] = "square"

    # Create environment
    # env = suite.make(
    env = suite.environments.manipulation.nut_assembly.NutAssemblySquare(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        #hard_reset=False,
    )

    save_datetime = datetime.now().strftime("%m_%d_%H_%M_%S")
    base_dir_home = "{}/demonstration_data".format(args.base_dir)
    tmp_dir = "{}/{}_{}".format(base_dir_home,
                                args.environment,
                                save_datetime)
    env = DataCollectionWrapper(env, tmp_dir)

    if args.checkpoint == "":
        policy = RandomPolicy(env)
    else:
        policy = TrainedPolicy(args.checkpoint)

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    save_dir = os.path.join(args.base_dir, "{}_iter_{}".format(save_datetime, args.training_iter))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = []

    data_fail = []

    start = time.perf_counter()

    total_human_samples = 0
    total_samples = 0
    agent_success = 0
    success_at_time_lst = []

    remove_directory = []

    env_info = json.dumps(config)

    for traj_id in tqdm(range(args.num_trajectories)):
        saving, this_human_sample, success_at_time, traj = collect_trajectory(env, device, args, data, remove_directory)
        print("saving: ", saving)
        print("success at time: ", success_at_time)
        if saving:
            total_human_samples += this_human_sample
            total_samples += len(traj["actions"])
            if this_human_sample == 0:
                agent_success += 1
                # print("AGENT SUCCESS: ", this_human_sample == 0)
            success_at_time_lst.append(success_at_time)
            gather_demonstrations_as_hdf5(tmp_dir, tmp_dir, env_info, remove_directory)

    end = time.perf_counter()

    """ Record Experiment Iteration Info """
    total_time = end - start
    num_success = len(data)
    success_rate = len(data) / args.num_trajectories
    agent_success_rate = agent_success / args.num_trajectories
    total_human_samples = total_human_samples
    total_samples = total_samples
    human_traj_ratio = total_human_samples / total_samples
    human_sample_per_traj = total_human_samples / len(data)
    aver_time_per_traj = total_time / args.num_trajectories
    aver_traj_length = total_samples / len(data)

    """ Save data into numpy array """
    save_name = "{}-iter_{}-n_{}_final.npy".format(args.environment,
                                                   args.training_iter,
                                                   len(data),
                                                   )

    save_name_failed = "{}-iter_{}-n_{}_final_failed.npy".format(args.environment,
                                                                 args.training_iter,
                                                                 len(data_fail),
                                                                 )

    print("total time: ", total_time)
    print("number of success: ", num_success)
    print("success rate: ", success_rate)
    print("agent success rate: ", agent_success_rate)
    print("total human samples: ", total_human_samples)
    print("total samples: ", total_samples)
    print("human traj ratio: ", human_traj_ratio)
    print("human sample per traj: ", human_sample_per_traj)
    print("aver time per traj: ", aver_time_per_traj)
    print("aver traj length: ", aver_traj_length)

    experiment_info = [
        args.training_iter,
        total_time,
        args.num_trajectories,
        num_success,
        success_rate,
        agent_success_rate,
        total_human_samples,
        total_samples,
        human_traj_ratio,
        human_sample_per_traj,
        aver_time_per_traj,
        aver_traj_length,
        args.checkpoint,
        os.path.join(save_dir, save_name),
    ]

    header = ["Training Round",
              "Total Time",
              "Number of Trajectories",
              "Number of Success",
              "Success Rate",
              "Policy Success Rate",
              "Total Human Samples",
              "Total Samples",
              "Human Sample Percentage",
              "Human Sample per Trajectory",
              "Average Time per Trajectory",
              "Average Trajectory Length",
              "Policy Rollouts Filename",
              "Checkpoint Used",
              "Save Name"
              ]

    from os.path import exists
    if not exists(args.csv_filename):
        write_header(args.csv_filename, header)
    write_to_csv(args.csv_filename, experiment_info)

    np.save(os.path.join(save_dir, save_name), data)
    np.save(os.path.join(save_dir, save_name_failed), data_fail)

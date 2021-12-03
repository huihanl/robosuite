"""Teleoperate robot with keyboard or SpaceMouse.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with Mac OS X, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports Mac OS X (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

    Notes:
        OSC also tends to be more computationally efficient since IK relies on the backend pybullet IK solver.


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note that the environments include sanity
        checks, such that a "TwoArm..." environment will only accept either a 2-tuple of robot names or a single
        bimanual robot name, according to the specified configuration (see below), and all other environments will
        only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"bimanual", "single-arm-parallel", and "single-arm-opposed"}

            -"bimanual": Sets up the environment for a single bimanual robot. Expects a single bimanual robot name to
                be specified in the --robots argument

            -"single-arm-parallel": Sets up the environment such that two single-armed robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of single-armed robot names to be specified
                in the --robots argument.

            -"single-arm-opposed": Sets up the environment such that two single-armed robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of single-armed robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-grasp: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config single-arm-parallel --controller osc


"""

import argparse
import numpy as np

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
from tqdm import tqdm
import time
import numpy as np
from datetime import datetime
import torch
import csv

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    parser.add_argument("--num-trajectories", type=int, default=20, help="Number of trajectories to collect / evaluate")
    parser.add_argument("--training-iter", type=int)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--csv-filename", type=str, default="")
    parser.add_argument("--all-demos", default=False, type=bool)
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
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    if args.environment == "NutAssembly":
        config["single_object_mode"] = 2
        config["nut_type"] = "square"

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

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

    data = []

    start = time.perf_counter()

    total_human_samples = 0
    total_samples = 0

    agent_success = 0

    for traj_id in tqdm(range(args.num_trajectories)):

        if len(data) % 5 == 0:

            save_name = "/home/huihanl/{}-iter_{}-n_{}_{}.npy".format(args.environment,
                                                        args.training_iter,
                                                        len(data), datetime.now().strftime("%m_%d_%H_%M_%S"))
            np.save(save_name, data)

        traj = dict(
            observations=[],
            actions=[],
            rewards=[],
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

        while True:

            # Set active robot
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

            # Get the newest action
            action, grasp = input2action(
                device=device,
                robot=active_robot,
                active_arm=args.arm,
                env_configuration=args.config
            )
            #print(action)
            if action is None:
                break

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

            #if human_take_control and is_empty_input_spacemouse(action):
            #    continue

            if is_empty_input_spacemouse(action):
                if args.all_demos:
                    continue # if all demos, no action
                else:
                    action = policy.get_action(obs["state"]) # if not all demos, use agent action
                #print(action)
                #continue
                is_human = 0
            else:
                this_human_sample += 1
                if args.all_demos:
                    is_human = 0 # iter 0 is viewed as non-intervention
                else:
                    is_human = 1
            #action += np.random.normal(0, 0.05, 7)
            # Step through the simulation and render
            next_obs, reward, done, info = env.step(action)

            next_obs = post_process_state(next_obs)

            traj["observations"].append(obs)
            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["next_observations"].append(next_obs)
            traj["terminals"].append(done)
            traj["is_human"].append(is_human)
            obs = next_obs
            # print("state obs: ")
            # print(traj["observations"])
            # print("action: ")
            # print(traj["actions"])

            env.render()

            if env._check_success():
                time_success += 1

            if time_success == 5:
                data.append(traj)
                print("trajectory length: ", len(traj["actions"]))
                total_human_samples += this_human_sample
                total_samples += len(traj["actions"])
                if this_human_sample == 0:
                    agent_success += 1
                    print("AGENT SUCCESS: ", this_human_sample == 0)
                break
            timestep_count += 1
            if timestep_count > 1600:
                print("discard this trial")
                break

    end = time.perf_counter()

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

    save_name = "/home/huihanl/{}-iter_{}-n_{}-{}_final.npy".format(args.environment,
                                                              args.training_iter,
                                                              len(data),
                                                              datetime.now().strftime("%m_%d_%H_%M_%S")
                                                              )

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
        save_name,
        args.checkpoint
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
              ]

    from os.path import exists
    if not exists(args.csv_filename):
        write_header(args.csv_filename, header)
    write_to_csv(args.csv_filename, experiment_info)

    np.save(save_name, data)

import os
import cv2
import yaml
import time
import torch
import wandb
import random
import shutil
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from offlinerl.algo.scq_discrete import SCQD
from offlinerl.algo.sac_discrete import SACD
from offlinerl.algo.sacn_discrete import SACND
from offlinerl.algo.cql_discrete import CQLD
from offlinerl.algo.td3bc_discrete import TD3BCD
from offlinerl.algo.iql_discrete import IQLD

from tqdm import tqdm
from arguments import get_args
from offlinerl.utils.replay_buffer import ReplayBuffer
from preprocessing import multi_frame_segment, video_to_frames
from automatic_search_env import AutomaticSearch, FrameSkipWrapper, SeedEnvWrapper


def parse_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def eval_policy(policy, traj_path, seed, args, eval_indicators, eval_episodes=10):
    eval_env = AutomaticSearch(is_FFT=args.is_FFT, is_SFJPD=args.is_SFJPD)
    eval_env = SeedEnvWrapper(eval_env, seed=(seed))

    if os.path.exists(traj_path) and os.path.isdir(traj_path):
        for filename in os.listdir(traj_path):
            file_path = os.path.join(traj_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print(f"Empty directory: {traj_path}")
    else:
        print(f"Directory does not exist: {traj_path}")

    avg_reward = 0.
    avg_step = 0
    avg_final_distance = 0.
    avg_success_rate = 0.
    avg_inference_time = 0.
    for dir_index in range(eval_episodes):
        state, _ = eval_env.reset(is_train=False)
        state_concat = np.concatenate((state, (state[0][:-2 * eval_env.sample_factor]).reshape(1, -1)), axis=1)
        for img_index in range(args.max_episode_steps):
            eval_env.save_trajectory(traj_path=traj_path, dir_index=str(dir_index), img_index=str(img_index))

            state_concat = np.array(state_concat).reshape(1, -1)
            start = time.time()
            action = policy.get_action(state_concat).reshape(-1)
            end = time.time()
            avg_inference_time += (end - start)

            next_state, reward, done, distance, _ = eval_env.step(action)
            state_concat = np.concatenate((state, (next_state[0][:-2 * eval_env.sample_factor]).reshape(1, -1)), axis=1)
            state = next_state

            avg_reward += reward
            avg_step += 1

            if done or img_index >= (args.max_episode_steps - 1):
                if done:
                    avg_success_rate += 1
                avg_final_distance += distance
                eval_env.save_trajectory(traj_path=traj_path, dir_index=str(dir_index), img_index=str(img_index))
                break

    avg_reward /= eval_episodes
    avg_inference_time /= avg_step
    avg_step /= eval_episodes
    avg_final_distance /= eval_episodes
    avg_success_rate /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    if eval_indicators[0] == 0 and eval_indicators[1] == 0:
        eval_indicators = [avg_final_distance, avg_step, avg_success_rate, avg_inference_time]
    else:
        eval_indicators = [(eval_indicators[0] + avg_final_distance) / 2, (eval_indicators[1] + avg_step) / 2,
                           (eval_indicators[2] + avg_success_rate) / 2, (eval_indicators[3] + avg_inference_time) / 2]

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} D4RL score: {d4rl_score[0]:.3f}, Average Final Pixel: {eval_indicators[0]},"
          f" Average Step: {eval_indicators[1]}, Average Success Rate: {eval_indicators[2]}, "
          f"Average Search Speed:{eval_indicators[3]}")
    print("---------------------------------------")
    return d4rl_score, avg_reward, avg_step, eval_indicators


if __name__ == "__main__":
    '''Split the complete video into pictures'''
    # video_dir = r'D:\OneDrive\Desktop\Paper_2\Birds_dataset\grus_japonensis'
    # output_dir = r'D:\Pycharm\Projects\automatic-search\datasets\train'
    #
    # video_paths = os.listdir(video_dir)
    # for video_path in video_paths:
    #     output_path = os.path.join(output_dir, os.path.splitext(video_path)[0])
    #     video_path = os.path.join(video_dir, video_path)
    #     print(video_path)
    #     video_to_frames(video_path, output_path, fps=15)

    '''segment images for preprocessing'''
    # dataset_dir = "./datasets/eval"
    # # video_dir_index = os.listdir(dataset_dir)
    # video_dir_index = ["8"]
    #
    # points = {}
    # labels = {}
    #
    # # test
    # labels["5"] = np.array([1], np.int32)
    # labels["6"] = np.array([1], np.int32)
    # labels["7"] = np.array([1], np.int32)
    # labels["8"] = np.array([1], np.int32)
    # labels["9"] = np.array([1, 1, 1, 1], np.int32)
    # labels["11"] = np.array([1], np.int32)
    # labels["12"] = np.array([1], np.int32)
    # labels["13"] = np.array([1], np.int32)
    #
    # points["5"] = np.array([[187, 513]], dtype=np.float32)  # [637, 376]
    # points["6"] = np.array([[604, 647]], dtype=np.float32)
    # points["7"] = np.array([[320, 600]], dtype=np.float32)
    # points["8"] = np.array([[503, 568]], dtype=np.float32)  # [515, 545], [510, 533],
    # points["9"] = np.array([[666, 545], [512, 570], [336, 535], [199, 529]], dtype=np.float32)
    # points["11"] = np.array([[486, 392]], dtype=np.float32) # [467, 602], [755, 418], [486, 392]
    # points["12"] = np.array([[533, 436]], dtype=np.float32)
    # points["13"] = np.array([[450, 458]], dtype=np.float32)
    #
    # '''
    # # Black Swan
    # labels["7"] = np.array([1, 0], np.int32)
    # labels["8"] = np.array([1, 0], np.int32)
    # labels["9"] = np.array([1, 0], np.int32)
    # labels["10"] = np.array([1, 0, 1], np.int32)
    # labels["11"] = np.array([1, 0], np.int32)
    # labels["12"] = np.array([1], np.int32)
    # labels["13"] = np.array([1], np.int32)
    # labels["14"] = np.array([1], np.int32)
    # points["7"] = np.array([[1246, 640], [753, 592]], dtype=np.float32)
    # points["8"] = np.array([[553, 622], [729, 632]], dtype=np.float32)
    # points["9"] = np.array([[497, 656], [626, 692]], dtype=np.float32)
    # points["10"] = np.array([[546, 539], [564, 568], [495, 564]], dtype=np.float32)
    # points["11"] = np.array([[478, 513], [235, 586]], dtype=np.float32)  # , [336, 531], [478, 513]
    # points["12"] = np.array([[783, 446]], dtype=np.float32)  # , [488, 437], [783, 446]
    # points["13"] = np.array([[616, 523]], dtype=np.float32)
    # points["14"] = np.array([[554, 498]], dtype=np.float32)
    #
    # # Gallinus nigra points and labels
    # # labels["1"] = np.array([1, 1, 0, 1, 1], np.int32)
    # # labels["2"] = np.array([1, 1, 0, 1, 1], np.int32)
    # # labels["3"] = np.array([1, 1, 0, 1, 1], np.int32)
    # # labels["4"] = np.array([1, 0, 1, 1], np.int32)
    # # labels["5"] = np.array([1, 0, 1, 1], np.int32)
    # # labels["6"] = np.array([1, 1, 0, 1, 1], np.int32)
    # # labels["7"] = np.array([1, 1, 0, 1, 1], np.int32)
    # # labels["8"] = np.array([1, 1, 0, 1, 1], np.int32)
    # # labels["9"] = np.array([1, 1, 1, 0, 1, 1], np.int32)
    # # labels["10"] = np.array([1, 1, 0, 1, 1], np.int32)
    # # labels["11"] = np.array([1, 1, 0, 1, 1], np.int32)
    # # points["1"] = np.array([[222, 512], [459, 525], [515, 513], [611, 528], [768, 527]], dtype=np.float32)
    # # points["2"] = np.array([[296, 583], [340, 566], [658, 626], [569, 589], [858, 582]], dtype=np.float32)
    # # points["3"] = np.array([[260, 554], [418, 578], [514, 592], [630, 566], [737, 589]], dtype=np.float32)
    # # points["4"] = np.array([[285, 646], [465, 654], [544, 621], [606, 617]], dtype=np.float32)
    # # points["5"] = np.array([[81, 643], [196, 639], [944, 607], [969, 605]], dtype=np.float32)
    # # points["6"] = np.array([[239, 595], [213, 590], [290, 594], [661, 574], [893, 565]], dtype=np.float32)
    # # points["7"] = np.array([[171, 659], [515, 651], [245, 662], [828, 639], [1245, 657]], dtype=np.float32)
    # # points["8"] = np.array([[171, 650], [360, 660], [527, 683], [618, 648], [981, 664]], dtype=np.float32)
    # # points["9"] = np.array([[174, 622], [181, 616], [402, 637], [533, 678], [608, 630], [869, 651]], dtype=np.float32)
    # # points["10"] = np.array([[316, 594], [376, 656], [642, 671], [512, 607], [560, 595]], dtype=np.float32)
    # # points["11"] = np.array([[302, 542], [430, 558], [520, 600], [663, 569], [713, 575]], dtype=np.float32)
    #
    # # seals points and labels
    # # labels["1"] = np.array([1], np.int32)
    # # labels["2"] = np.array([1], np.int32)
    # # labels["3"] = np.array([11], np.int32)
    # # labels["4"] = np.array([1], np.int32)
    # # labels["5"] = np.array([1], np.int32)
    # # labels["6"] = np.array([1], np.int32)
    # # labels["7"] = np.array([1], np.int32)
    # # points["1"] = np.array([[360, 910]], dtype=np.float32)
    # # points["2"] = np.array([[730, 1070]], dtype=np.float32)
    # # points["3"] = np.array([[806, 944]], dtype=np.float32)
    # # points["4"] = np.array([[882, 600]], dtype=np.float32)
    # # points["5"] = np.array([[584, 1380]], dtype=np.float32)
    # # points["6"] = np.array([[1260, 1120]], dtype=np.float32)
    # # points["7"] = np.array([[1300, 1220]], dtype=np.float32)
    # '''
    #
    # for index in video_dir_index:
    #     video_dir = os.path.join(dataset_dir, index)
    #
    #     frame_names = [
    #         p for p in os.listdir(video_dir)
    #         if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    #     ]
    #     frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    #
    #     multi_frame_segment(video_dir, frame_names, points[index], labels[index], task=index)

    '''Collect offline data from simulator'''
    args = get_args()
    args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    if args.config:
        config = parse_yaml(args.config)

        for key, value in config.items():
            setattr(args, key, value)

    env = AutomaticSearch(is_FFT=args.is_FFT, is_SFJPD=args.is_SFJPD)
    # set seed
    env = SeedEnvWrapper(env, seed=args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    training_data_path = r'./datasets/train/data.h5'
    state_dim = env.observation_space
    action_dim = env.action_space.n
    max_action = float(env.action_space.n - 1)

    # data operation
    if (args.is_FFT and not args.is_SFJPD) or (not args.is_FFT and not args.is_SFJPD):
        state_dim = state_dim - 2 * env.sample_factor
    else:
        state_dim = state_dim * 2 - 2 * env.sample_factor

    replay_buffer = ReplayBuffer(state_dim, action_dim, args.device)
    alpha = 0.8
    counter = 0
    for episode in tqdm(range(args.max_episodes)):
        obs, info = env.reset()
        state = np.concatenate((obs, (obs[0][:-2 * env.sample_factor]).reshape(1, -1)), axis=1)
        action = env.action_space.sample()
        for t in range(args.max_episode_steps):
            # env.save_trajectory(traj_path=r'/home/wht/automatic-search/results/test',
            #                 dir_index=str(episode), img_index=str(replay_buffer.size))

            next_obs, reward, done, _, _ = env.step(action)
            counter += 1

            next_state = np.concatenate((obs, (next_obs[0][:-2 * env.sample_factor]).reshape(1, -1)), axis=1)
            replay_buffer.add(state, action, next_state, reward, done)
            obs = next_obs
            state = next_state

            if np.random.uniform(0, 1) >= alpha:
                action = env.action_space.sample()
            else:
                current_pos = np.array(env.current_pos)
                goal_pos = np.array(env.goal_pos)
                distances = np.linalg.norm(goal_pos - current_pos, axis=1)
                nearest_index = np.argmin(distances)

                distance = goal_pos[nearest_index] - current_pos

                if abs(distance[0]) > abs(distance[1]):
                    if distance[0] < 0:
                        action = 3
                    else:
                        action = 4
                else:
                    if distance[1] < 0:
                        action = 1
                    else:
                        action = 2

            if done:
                # env.env.save_trajectory(traj_path=r"/home/wht/automatic-search/results/", dir_index=str("test"), img_index=str(counter))
                # env.save_trajectory(traj_path=r'/home/wht/automatic-search/results/test',
                #                     dir_index=str(episode), img_index=str(replay_buffer.size))
                cv2.waitKey(0)
                break

    print(replay_buffer.size)

    '''train RL agent'''
    if args.policy == "SCQD":
        policy = SCQD(state_dim, action_dim, max_action, args)
    elif args.policy == "SACD":
        policy = SACD(state_dim, action_dim, max_action, args)
    elif args.policy == "SACND":
        policy = SACND(state_dim, action_dim, max_action, args)
    elif args.policy == "CQLD":
        policy = CQLD(state_dim, action_dim, max_action, args)
    elif args.policy == "TD3BCD":
        policy = TD3BCD(state_dim, action_dim, max_action, args)
    elif args.policy == "IQLD":
        policy = TD3BCD(state_dim, action_dim, max_action, args)
    else:
        raise NotImplementedError

    project_name = args.env
    group_name = args.env
    if args.option_name:
        group_name = group_name + "_" + str(args.option_name)

    wandb.init(project="Agent search animals automaticly", config=args, group=group_name)
    if args.lagrange_tau:
        wandb.run.name = f"{project_name}_lagrange_tau{args.lagrange_tau}_seed{args.seed}"
    else:
        wandb.run.name = f"{project_name}_lam{args.critic_penalty_coef}_seed{args.seed}"

    if args.actor_penalty_coef:
        wandb.run.name = wandb.run.name + "_actor_lam:" + str(args.actor_penalty_coef)

    if args.option_name:
        wandb.run.name = wandb.run.name + "_" + str(args.option_name)

    wandb.mark_preempting()

    eval_indicators = [0., 0., 0., 0.]
    max_time_steps = int(args.max_timesteps)
    for step in tqdm(range(max_time_steps)):
        state, action, next_state, reward, done = replay_buffer.sample(args.batch_size)

        # train policy
        policy.train(state, action, next_state, reward, done, step)

        # Evaluate episode
        if step % args.eval_freq == 0 or step == max_time_steps - 1:
            d4rl_score, avg_reward, avg_step, eval_indicators = eval_policy(policy, args.trajectory_path, args.seed,
                                                                            args, eval_indicators,
                                                                            eval_episodes=args.eval_episodes)
            if args.checkpoints_path:
                if not os.path.exists(args.checkpoints_path):
                    os.makedirs(args.checkpoints_path)

                # torch.save(
                #     {"vae": policy.vae,
                #      "actor": policy.actor,
                #      "critic1": policy.critic1,
                #      "critic2": policy.critic2,
                #      "critic1_target": policy.critic1_target,
                #      "critic2_target": policy.critic2_target,
                #      "critic1_opt": policy.critic1_opt,
                #      "critic2_opt": policy.critic2_opt,
                #      },
                #     os.path.join(args.checkpoints_path, f"checkpoint_{step}.pt")
                # )

                torch.save(
                    {"actor": policy.actor
                     },
                    os.path.join(args.checkpoints_path, f"checkpoint_{step}.pt")
                )

            wandb.log({"eval/step": step,
                       "eval/d4rl_score": d4rl_score[0],
                       "eval/return": avg_reward[0],
                       "eval/episode length": avg_step, })
    wandb.finish()

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--config", default='configs/automatic_search.yaml', type=str,
                        help='Path to the configuration file', required=True)
    parser.add_argument("--checkpoints_path", type=str, help='Path to the checkpoints')
    parser.add_argument("--trajectory_path", type=str, help='Path to the trajectory')
    parser.add_argument("--seed", default=10, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--log_interval", default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_episodes", default=10, type=int)  # Number of trajectories for evaluation
    parser.add_argument("--option_name", default=None)

    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--actor_learning_rate", default=3e-4, type=float)  # Actor learning rate
    parser.add_argument("--critic_learning_rate", default=3e-4, type=float)  # Critic learning rate
    parser.add_argument("--soft_target_tau", default=0.005)  # Target network update rate
    parser.add_argument("--actor_num_hidden_layers", default=2, type=int)
    parser.add_argument("--critic_num_hidden_layers", default=2, type=int)
    parser.add_argument("--hidden_layer_dim", default=512, type=int)
    parser.add_argument("--actor_clip_grad_norm", default=None, type=float)
    parser.add_argument("--critic_clip_grad_norm", default=None, type=float)
    parser.add_argument("--data_size_ratio", default=None, type=int)  # data size ratio for experiment

    parser.add_argument("--actor_penalty_coef", default=None, type=float)
    parser.add_argument("--critic_penalty_coef", default=0.0, type=float)
    parser.add_argument("--normalize_reward", default=False)
    parser.add_argument("--use_layernormalization", default=False)
    parser.add_argument("--use_actor_scheduler", default=False)
    parser.add_argument("--log_sig_min", default=-5.0, type=float)
    parser.add_argument("--log_sig_max", default=2.0, type=float)
    parser.add_argument("--vae_hidden_layer_dim", default=1024, type=int)
    parser.add_argument("--vae_num_hidden_layers", default=1, type=int)
    parser.add_argument("--vae_learning_rate", default=1e-3, type=float)
    parser.add_argument("--vae_sampling_num", default=10, type=int)

    parser.add_argument("--use_automatic_entropy_tuning", default=True)
    parser.add_argument("--lagrange_tau", default=None, type=float)
    args = parser.parse_args()

    return args

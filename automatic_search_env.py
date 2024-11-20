import os
import cv2
import json
import torch
import random
import gymnasium
import numpy as np

from typing import Optional
from collections import deque
import torch.nn.functional as F
from offlinerl.algo.cvae import ConditionalVAE
from skimage.metrics import structural_similarity as ssim


class AutomaticSearch(gymnasium.Env):
    def __init__(self, is_FFT=False, is_SFJPD=False):
        super(AutomaticSearch, self).__init__()

        self.device = 'cuda:0'
        self.train_dataset_dir = os.path.join(os.getcwd(), 'datasets/train')
        self.eval_dataset_dir = os.path.join(os.getcwd(), 'datasets/eval')
        self.image_encoder_path = os.path.join(os.getcwd(), 'checkpoints/VAE/best_4.pt')

        self.vae_model = None
        self.image = None
        self.current_obs = None
        self.current_pos = np.zeros(2)

        self.goal_obs = []
        self.goal_pos = []

        self.min_x, self.min_y = 64, 64
        self.max_x, self.max_y = 1216, 656

        self.max_distance = 608 + 328
        self.max_offset = 32

        if (is_FFT and is_SFJPD) or (not is_FFT and is_SFJPD):
            self.sample_factor = 60
        else:
            self.sample_factor = 0

        self.step_count = 0
        self.dir_index = 1
        self.image_index = 0
        self.identification_bit = 0

        self.is_FFT = is_FFT
        self.is_SFJPD = is_SFJPD

    @property
    def observation_space(self):
        if self.is_FFT and self.is_SFJPD:
            return 512 * 4 * 4 + 3 * self.sample_factor
        elif self.is_FFT and not self.is_SFJPD:
            return 512 * 4 * 4
        elif self.is_SFJPD and not self.is_FFT:
            return 128 * 128 * 3 + 3 * self.sample_factor
        else:
            return 128 * 128 * 3

    # 1=up 2=down 3=left 4=right
    @property
    def action_space(self):
        return gymnasium.spaces.Discrete(5)

    def get_normalized_score(self, avg_reward):
        return avg_reward / (self.max_offset * 5)

    def step(self, action):
        action = np.array(action)
        assert action >= 0, action

        done = False
        self.step_count += 1

        if action == 1:  # up
            self.current_pos[1] -= 64
        elif action == 2:  # down
            self.current_pos[1] += 64
        elif action == 3:  # left
            self.current_pos[0] -= 64
        elif action == 4:  # right
            self.current_pos[0] += 64
        else:  # none
            pass

        self.current_pos[0] = np.clip(self.current_pos[0], self.min_x, self.max_x)
        self.current_pos[1] = np.clip(self.current_pos[1], self.min_y, self.max_y)

        self.current_obs = self.clip_obs(self.current_pos[0], self.current_pos[1])

        reward = self.compute_reward()
        distance, done = self.compute_distance()

        if self.is_FFT and self.is_SFJPD:
            latent_image = self.vae_model.encoder(
                torch.FloatTensor(self.current_obs).permute(2, 0, 1).unsqueeze(0).to(self.device))
            latent_image = latent_image.view(-1, self.observation_space - 3 * self.sample_factor).cpu().detach().numpy()
            latent_state = np.concatenate((self.current_pos.reshape(1, 2), np.array([[self.identification_bit]])),
                                          axis=1)
            latent_state = self.vae_model.fc_state(torch.FloatTensor(latent_state).to(self.device))
            latent_state = F.interpolate(latent_state.unsqueeze(2),
                                         scale_factor=self.sample_factor, mode='linear', align_corners=True)
            latent_state = (F.normalize(latent_state, p=2, dim=1)).squeeze().view(-1,
                                                                                  3 * self.sample_factor).cpu().detach().numpy()
            latent = np.concatenate((latent_image, latent_state), axis=1)
        elif self.is_FFT and not self.is_SFJPD:
            latent_image = self.vae_model.encoder(
                torch.FloatTensor(self.current_obs).permute(2, 0, 1).unsqueeze(0).to(self.device))
            latent = latent_image.view(-1, self.observation_space).cpu().detach().numpy()
        elif self.is_SFJPD and not self.is_FFT:
            latent_image = torch.FloatTensor(self.current_obs.flatten()).to(self.device).unsqueeze(0)
            latent_image = latent_image.view(-1, self.observation_space - 3 * self.sample_factor).cpu().detach().numpy()
            latent_state = np.concatenate((self.current_pos.reshape(1, 2), np.array([[self.identification_bit]])),
                                          axis=1)
            latent_state = self.vae_model.fc_state(torch.FloatTensor(latent_state).to(self.device))
            latent_state = F.interpolate(latent_state.unsqueeze(2),
                                         scale_factor=self.sample_factor, mode='linear', align_corners=True)
            latent_state = (F.normalize(latent_state, p=2, dim=1)).squeeze().view(-1,
                                                                                  3 * self.sample_factor).cpu().detach().numpy()
            latent = np.concatenate((latent_image, latent_state), axis=1)
        else:
            latent_image = torch.FloatTensor(self.current_obs.flatten()).to(self.device).unsqueeze(0)
            latent = latent_image.view(-1, self.observation_space).cpu().detach().numpy()

        return latent, np.array([reward]), np.array([done]), distance, {}
        # return np.transpose(self.current_obs, (2, 0, 1)), np.array([reward]), np.array([done]), distance, {}

    def reset(self, is_train=True, seed: Optional[int] = None):
        # File operation
        if is_train:
            dir_index = self.dir_index
            images_path = os.path.join(self.train_dataset_dir, str(dir_index))

            if self.image_index >= len(os.listdir(images_path)) - 1:
                self.dir_index = (dir_index + 1) % (len(os.listdir(self.train_dataset_dir)) + 1)
                dir_index = self.dir_index
                images_path = os.path.join(self.train_dataset_dir, str(dir_index))
                self.image_index = 0

            dataset_dir = self.train_dataset_dir
            image_index = self.image_index
            self.image_index = (self.image_index + 1) % len(os.listdir(images_path))

            self.max_offset = 50
        else:
            dir_index = np.random.randint(1, len(os.listdir(self.eval_dataset_dir)) + 1)
            images_path = os.path.join(self.eval_dataset_dir, str(dir_index))
            dataset_dir = self.eval_dataset_dir
            image_index = np.random.randint(0, len(os.listdir(images_path)) - 1)

            self.max_offset = 50

        image = cv2.imread(os.path.join(images_path, f"{image_index:05d}.jpg"))
        self.image = image

        with open(os.path.join(dataset_dir, str(dir_index) + "/images_info.json"), "r") as f:
            goal_info = json.load(f)
            goal_info_dict = goal_info[image_index]
            goal_centers = goal_info_dict['centers']
            self.goal_pos = goal_centers

            for central_pos in self.goal_pos:
                self.goal_obs.append([self.clip_obs(central_pos[0], central_pos[1])])

        # Randomly crop the complete image into obs
        if is_train:
            random_x, random_y = random.randint(self.min_x, self.max_x), random.randint(self.min_y, self.max_y)
        else:
            random_x, random_y = self.min_x, self.min_y
        obs = self.clip_obs(random_x, random_y)

        self.current_pos = np.array([random_x, random_y])
        self.current_obs = obs

        if self.is_FFT and self.is_SFJPD:
            # load cvae model
            checkpoint = torch.load(self.image_encoder_path, map_location=self.device)
            self.vae_model = ConditionalVAE(
                in_channels=3,
                latent_dim=256,
                hidden_dims=None,
                img_size=128
            ).to(self.device)
            self.vae_model.encoder.load_state_dict(checkpoint['encoder'].state_dict())
            self.vae_model.fc_state.load_state_dict(checkpoint['fc_state'].state_dict())

            self.step_count = 0
            self.identification_bit = 0
            latent_image = self.vae_model.encoder(torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device))
            latent_image = latent_image.view(-1, self.observation_space - 3 * self.sample_factor).cpu().detach().numpy()
            latent_state = np.concatenate((self.current_pos.reshape(1, 2), np.array([[self.identification_bit]])),
                                          axis=1)
            latent_state = self.vae_model.fc_state(torch.FloatTensor(latent_state).to(self.device))
            latent_state = F.interpolate(latent_state.unsqueeze(2),
                                         scale_factor=self.sample_factor, mode='linear', align_corners=True)
            latent_state = (F.normalize(latent_state, p=2, dim=1)).squeeze().view(-1,
                                                                                  3 * self.sample_factor).cpu().detach().numpy()
            latent = np.concatenate((latent_image, latent_state), axis=1)
        elif self.is_FFT and not self.is_SFJPD:
            # load cvae model
            checkpoint = torch.load(self.image_encoder_path, map_location=self.device)
            self.vae_model = ConditionalVAE(
                in_channels=3,
                latent_dim=256,
                hidden_dims=None,
                img_size=128
            ).to(self.device)
            self.vae_model.encoder.load_state_dict(checkpoint['encoder'].state_dict())

            self.step_count = 0
            self.identification_bit = 0
            latent_image = self.vae_model.encoder(torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device))
            latent = latent_image.view(-1, self.observation_space).cpu().detach().numpy()
        elif self.is_SFJPD and not self.is_FFT:
            # load cvae model
            checkpoint = torch.load(self.image_encoder_path, map_location=self.device)
            self.vae_model = ConditionalVAE(
                in_channels=3,
                latent_dim=256,
                hidden_dims=None,
                img_size=128
            ).to(self.device)
            self.vae_model.fc_state.load_state_dict(checkpoint['fc_state'].state_dict())

            self.step_count = 0
            self.identification_bit = 0
            latent_image = torch.FloatTensor(obs.flatten()).to(self.device).unsqueeze(0)
            latent_image = latent_image.view(-1, self.observation_space - 3 * self.sample_factor).cpu().detach().numpy()
            latent_state = np.concatenate((self.current_pos.reshape(1, 2), np.array([[self.identification_bit]])),
                                          axis=1)
            latent_state = self.vae_model.fc_state(torch.FloatTensor(latent_state).to(self.device))
            latent_state = F.interpolate(latent_state.unsqueeze(2),
                                         scale_factor=self.sample_factor, mode='linear', align_corners=True)
            latent_state = (F.normalize(latent_state, p=2, dim=1)).squeeze().view(-1,
                                                                                  3 * self.sample_factor).cpu().detach().numpy()
            latent = np.concatenate((latent_image, latent_state), axis=1)
        else:
            latent_image = torch.FloatTensor(obs.flatten()).to(self.device).unsqueeze(0)
            latent = latent_image.view(-1, self.observation_space).cpu().detach().numpy()

        return latent, [dir_index, image_index]
        # return np.transpose(obs, (2, 0, 1)), [dir_index, image_index]

    def compute_distance(self):
        distance = []
        is_done = False
        self.identification_bit = 0

        for goal_pos in self.goal_pos:
            distance.append(abs(np.sum(goal_pos - self.current_pos)))

            if np.all(abs(goal_pos - self.current_pos) < self.max_offset):
                self.identification_bit = 1
                is_done = True

        distance = min(distance)

        return distance, is_done

    def compute_reward(self):
        distance, _ = self.compute_distance()

        if distance > 0:
            reward_distance = self.max_offset * 2 / distance
        else:
            reward_distance = self.max_offset * 2

        # image1, image2 = self.current_obs, self.goal_obs
        # max_height = max(image1.shape[0], image2.shape[0])
        # max_width = max(image1.shape[1], image2.shape[1])
        # image1_padded = cv2.copyMakeBorder(image1, 0, max_height - image1.shape[0], 0, max_width - image1.shape[1],
        #                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # image2_padded = cv2.copyMakeBorder(image2, 0, max_height - image2.shape[0], 0, max_width - image2.shape[1],
        #                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # reward_similarity = self.calculate_image_similarity(image1_padded, image2_padded) * 5

        reward = np.clip(reward_distance, 0, self.max_offset * 2)

        return reward

    def clip_obs(self, pos_x, pox_y):
        start_x = pos_x - 64
        start_y = pox_y - 64
        end_x = pos_x + 64
        end_y = pox_y + 64
        obs = self.image[start_y:end_y, start_x:end_x]

        return np.array(obs)

    def save_trajectory(self, traj_path, dir_index, img_index):
        save_path = os.path.join(traj_path, dir_index)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(os.path.join(save_path, f"{img_index}.jpg"), self.current_obs)

    def calculate_image_similarity(self, image1, image2):
        image1 = cv2.resize(image1, (128, 128))
        image2 = cv2.resize(image2, (128, 128))

        grayA = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(grayA, grayB, full=True)
        return score

    def update_setting(self):
        assert self.image is not None

        self.min_x, self.min_y, self.min_z = self.current_pos[2] // 2, self.current_pos[2] // 2, 64
        self.max_x, self.max_y, self.max_z = self.image.shape[1] - self.current_pos[0] // 2, self.image.shape[0] - \
                                             self.current_pos[1] // 2, 256

        self.max_distance = (self.max_x - self.min_x) + (self.max_y - self.min_y) + (self.max_z - self.min_z)

    def setting_backtracking(self, pos, obs):
        self.current_pos = pos
        self.current_obs = obs


class FrameSkipWrapper(gymnasium.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        total_reward = 0
        self.obs_buffer = deque(maxlen=4)
        for t in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)

            if not t:
                restore_pos = self.env.current_pos

            self.obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        if len(self.obs_buffer) == 4:
            # obs = np.max(np.stack(self.obs_buffer), axis=0)
            obs = self.obs_buffer
        else:
            # obs = self.obs_buffer[0]
            for _ in range(4 - len(self.obs_buffer)):
                self.obs_buffer.append(obs)

            obs = self.obs_buffer

        self.env.setting_backtracking(restore_pos, self.obs_buffer[0])

        return obs, total_reward, done, truncated, info


class SeedEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed
        self.env.action_space.seed(seed)

    def reset(self, **kwargs):
        kwargs["seed"] = self.seed
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        return self.env.step(action)
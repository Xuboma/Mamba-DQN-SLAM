from typing import Callable, Union

import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F

from model.dtqn.agents.drqn import DqnAgent
from model.dtqn.agents.dqn import TrainMode
from Utils.bag import Bag
from Utils.context import Context
from Utils.random import RNG
from numpy.random import Generator, PCG64

from torchvision import models, transforms
from torchvision.models import mobilenet_v2

def process_image_with_mobilenet(image, model):
    """
    使用预训练的 MobileNet 模型处理图像
    :param image: 需要处理的图像
    :param model: 预训练的 MobileNet 模型
    :return: 处理后的图像
    """
    # 调整图像的大小，然后转换为 Tensor
    image = image.resize((224, 224))  # 注意，MobileNet V2 需要的输入大小是 224x224
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)

    # 确保张量是浮点型
    image = image.float()

    # 预处理图像
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    input_tensor = preprocess(image)
    #下面是euroc使用
    # image = image.resize((224, 224))  # 注意，MobileNet V2 需要的输入大小是 224x224
    # image = transforms.ToTensor()(image)
    # image = image.unsqueeze(0)
    #
    # # 预处理图像
    # preprocess = transforms.Compose([
    #     transforms.Normalize(mean=[0.485], std=[0.229])
    # ])
    # input_tensor = preprocess(image)

    # 使用 MobileNet 模型处理图像
    query_features = model(input_tensor)

    # 将特征重塑为64*64的大小
    query_features = query_features.view(-1, 64, 64)

    # 移除维度大小为1的维度
    query_features = query_features.squeeze(0)

    # 将结果转换为numpy数组
    processed_image = query_features.detach().numpy()

    return processed_image

class DtqnAgent(DqnAgent):
    # noinspection PyTypeChecker
    def __init__(
        self,
        network_factory: Callable[[], Module],
        buffer_size: int,
        device: torch.device,
        env_obs_length: int,
        max_env_steps: int,
        obs_mask: Union[int, float],
        num_actions: int,
        is_discrete_env: bool,
        learning_rate: float = 0.0003,
        batch_size: int = 16,
        context_len: int = 50,
        gamma: float = 0.99,
        grad_norm_clip: float = 1.0,
        target_update_frequency: int = 10_000,
        history: int = 50,
        bag_size: int = 0,
        **kwargs,
    ):
        super().__init__(
            network_factory,
            buffer_size,
            device,
            env_obs_length,
            max_env_steps,
            obs_mask,
            num_actions,
            is_discrete_env,
            learning_rate,
            batch_size,
            context_len,
            gamma,
            grad_norm_clip,
            target_update_frequency,
        )
        self.model = mobilenet_v2(pretrained=True)
        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                    bias=False)  # 修改第一层卷积的输入通道数为1
        self.model.classifier[1] = torch.nn.Linear(1280, 64 * 64)  # 添加一个额外的全连接层来将输出尺寸调整为64*64
        self.model.eval()

        self.history = history
        self.train_context = Context(
            context_len,
            obs_mask,
            num_actions,
            env_obs_length,
        )
        # print(num_actions)
        self.eval_context = Context(
            context_len,
            obs_mask,
            num_actions,
            env_obs_length,
        )

        self.train_bag = Bag(bag_size, obs_mask, env_obs_length)
        self.eval_bag = Bag(bag_size, obs_mask, env_obs_length)

    @property
    def bag(self) -> Bag:
        if self.train_mode == TrainMode.TRAIN:
            return self.train_bag
        elif self.train_mode == TrainMode.EVAL:
            return self.eval_bag

    @torch.no_grad()
    def get_action(self, epsilon: float = 0.0) -> int:
        if RNG.rng.random() <= epsilon:
            return RNG.rng.integers(self.num_actions)
        # Truncate the context of observations and actions to remove padding if it exists
        context_obs_tensor = torch.as_tensor(
            self.context.obs[: min(self.context.max_length, self.context.timestep + 1)],
            dtype=self.obs_tensor_type,
            device=self.device,
        ).unsqueeze(0)
        context_action_tensor = torch.as_tensor(
            self.context.action[
                : min(self.context.max_length, self.context.timestep + 1)
            ],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        # Always include the full bag, even if it has padding TODO:
        bag_obs_tensor = torch.as_tensor(
            self.bag.obss, dtype=self.obs_tensor_type, device=self.device
        ).unsqueeze(0)
        bag_action_tensor = torch.as_tensor(
            self.bag.actions, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        # print(self.context.action)
        q_values = self.policy_network(
            context_obs_tensor, context_action_tensor, bag_obs_tensor, bag_action_tensor
        )

        # We take the argmax of the last timestep's Q values
        # In other words, select the highest q value action
        # print(torch.argmax(q_values[:, -1, :]).item())
        return torch.argmax(q_values[:, -1, :]).item()

    def context_reset(self, obs: np.ndarray) -> None:
        self.context.reset(obs)
        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store_obs(obs)
        if self.bag.size > 0:
            self.bag.reset()


    def observe(self, obs: np.ndarray, action: int, reward: float, done: bool) -> None:
        """Add an observation to the context. If the context would evict an observation to make room,
        attempt to put the observation in the bag, which may require evicting something else from the bag.

        If we're in train mode, then we also add the transition to our replay buffer."""
        obs = process_image_with_mobilenet(obs, self.model)
        print("照片提取特征中")
        evicted_obs, evicted_action = self.context.add_transition(
            obs, action, reward, done
        )
        # print(self.context)
        # If there is an evicted obs, we need to decide if it should go in the bag or not
        if self.bag.size > 0 and evicted_obs is not None:
            # Bag is already full
            if not self.bag.add(evicted_obs, evicted_action):
                # For each possible bag, get the Q-values
                possible_bag_obss = np.tile(self.bag.obss, (self.bag.size + 1, 1, 1))
                possible_bag_actions = np.tile(
                    self.bag.actions, (self.bag.size + 1, 1, 1)
                )
                for i in range(self.bag.size):
                    evicted_obs = evicted_obs.reshape(-1)
                    possible_bag_obss[i, i] = evicted_obs
                    possible_bag_actions[i, i] = evicted_action
                tiled_context = np.tile(self.context.obs, (self.bag.size + 1, 1, 1))
                tiled_actions = np.tile(self.context.action, (self.bag.size + 1, 1, 1))
                q_values = self.policy_network(
                    torch.as_tensor(
                        tiled_context, dtype=self.obs_tensor_type, device=self.device
                    ),
                    torch.as_tensor(
                        tiled_actions, dtype=torch.long, device=self.device
                    ),
                    torch.as_tensor(
                        possible_bag_obss,
                        dtype=self.obs_tensor_type,
                        device=self.device,
                    ),
                    torch.as_tensor(
                        possible_bag_actions, dtype=torch.long, device=self.device
                    ),
                )

                bag_idx = torch.argmax(torch.mean(torch.max(q_values, 2)[0], 1))
                self.bag.obss = possible_bag_obss[bag_idx]
                self.bag.actions = possible_bag_actions[bag_idx]

        if self.train_mode == TrainMode.TRAIN:
            self.replay_buffer.store(obs, action, reward, done, self.context.timestep)


    # def train(self) -> None:
    #     if not self.replay_buffer.can_sample(self.batch_size):
    #         # print(self.batch_size)
    #         return
    #     self.eval_off()
    #     if self.bag.size > 0:
    #         (
    #             obss,
    #             actions,
    #             rewards,
    #             next_obss,
    #             next_actions,
    #             dones,
    #             episode_lengths,
    #             bag_obss,
    #             bag_actions,
    #         ) = self.replay_buffer.sample_with_bag(self.batch_size, self.bag)
    #         # Bags: [batch-size x bag-size x obs-dim]
    #         bag_obss = torch.as_tensor(
    #             bag_obss, dtype=self.obs_tensor_type, device=self.device
    #         )
    #         bag_actions = torch.as_tensor(
    #             bag_actions, dtype=torch.long, device=self.device
    #         )
    #     else:
    #         (
    #             obss,
    #             actions,
    #             rewards,
    #             next_obss,
    #             next_actions,
    #             dones,
    #             episode_lengths,
    #         ) = self.replay_buffer.sample(self.batch_size)
    #         # print(self.replay_buffer.max_size)
    #         bag_obss = None
    #         bag_actions = None
    #
    #     # Obss and Next obss: [batch-size x hist-len x obs-dim]
    #     obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
    #     next_obss = torch.as_tensor(
    #         next_obss, dtype=self.obs_tensor_type, device=self.device
    #     )
    #     # Actions: [batch-size x hist-len x 1]
    #     actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
    #     next_actions = torch.as_tensor(
    #         next_actions, dtype=torch.long, device=self.device
    #     )
    #     # Rewards: [batch-size x hist-len x 1]
    #     rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
    #     # Dones: [batch-size x hist-len x 1]
    #     dones = torch.as_tensor(dones, dtype=torch.long, device=self.device)
    #
    #     # obss is [batch-size x hist-len x obs-len]
    #     # then q_values is [batch-size x hist-len x n-actions]
    #     q_values = self.policy_network(obss, actions, bag_obss, bag_actions)
    #
    #     # After gathering, Q values becomes [batch-size x hist-len x 1] then
    #     # after squeeze becomes [batch-size x hist-len]
    #     q_values = q_values.gather(2, actions).squeeze()
    #
    #     with torch.no_grad():
    #         # Next obss goes from [batch-size x hist-len x obs-dim] to
    #         # [batch-size x hist-len x n-actions] and then goes through gather and squeeze
    #         # to become [batch-size x hist-len]
    #         if self.history:
    #             argmax = torch.argmax(
    #                 self.policy_network(next_obss, next_actions, bag_obss, bag_actions),
    #                 dim=2,
    #             ).unsqueeze(-1)
    #             next_obs_q_values = self.target_network(
    #                 next_obss, next_actions, bag_obss, bag_actions
    #             )
    #             next_obs_q_values = next_obs_q_values.gather(2, argmax).squeeze()
    #
    #         # here goes BELLMAN
    #         targets = rewards.squeeze() + (1 - dones.squeeze()) * (
    #             next_obs_q_values * self.gamma
    #         )
    #
    #     q_values = q_values[:, -self.history :]
    #     targets = targets[:, -self.history :]
    #     # Calculate loss
    #     loss = F.mse_loss(q_values, targets)
    #     # Log Losses
    #     self.qvalue_max.add(q_values.max().item())
    #     self.qvalue_mean.add(q_values.mean().item())
    #     self.qvalue_min.add(q_values.min().item())
    #
    #     self.target_max.add(targets.max().item())
    #     self.target_mean.add(targets.mean().item())
    #     self.target_min.add(targets.min().item())
    #
    #     self.td_errors.add(loss.item())
    #     # Optimization step
    #     self.optimizer.zero_grad(set_to_none=True)
    #     loss.backward()
    #     norm = torch.nn.utils.clip_grad_norm_(
    #         self.policy_network.parameters(),
    #         self.grad_norm_clip,
    #         error_if_nonfinite=True,
    #     )
    #     # Logging
    #     self.grad_norms.add(norm.item())
    #
    #     self.optimizer.step()
    #     self.num_train_steps += 1
    #
    #     if self.num_train_steps % self.target_update_frequency == 0:
    #         self.target_update()


    def train(self) -> None:
        print("开始网络训练")
        self.eval_off()
        # 用不到bag
        if self.bag.size > 0:
            (
                obss,
                actions,
                rewards,
                next_obss,
                next_actions,
                dones,
                episode_lengths,
                bag_obss,
                bag_actions,
            ) = self.replay_buffer.sample_with_bag(self.batch_size, self.bag)
            # Bags: [batch-size x bag-size x obs-dim]
            bag_obss = torch.as_tensor(
                bag_obss, dtype=self.obs_tensor_type, device=self.device
            )
            bag_actions = torch.as_tensor(
                bag_actions, dtype=torch.long, device=self.device
            )
        else: #直接从这里开始运行
            (
                obss,
                actions,
                rewards,
                next_obss,
                next_actions,
                dones,
                episode_lengths,
            ) = self.replay_buffer.sample(self.batch_size)
            # print(self.replay_buffer.max_size)
            bag_obss = None
            bag_actions = None

        # Obss and Next obss: [batch-size x hist-len x obs-dim]
        # print(actions)
        obss = torch.as_tensor(obss, dtype=self.obs_tensor_type, device=self.device)
        next_obss = torch.as_tensor(
            next_obss, dtype=self.obs_tensor_type, device=self.device
        )
        # Actions: [batch-size x hist-len x 1]
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        next_actions = torch.as_tensor(
            next_actions, dtype=torch.long, device=self.device
        )
        # Rewards: [batch-size x hist-len x 1]
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        # Dones: [batch-size x hist-len x 1]
        dones = torch.as_tensor(dones, dtype=torch.long, device=self.device)

        # obss is [batch-size x hist-len x obs-len]
        # then q_values is [batch-size x hist-len x n-actions]
        q_values = self.policy_network(obss, actions, bag_obss, bag_actions)

        # After gathering, Q values becomes [batch-size x hist-len x 1] then
        # after squeeze becomes [batch-size x hist-len]
        q_values = q_values.gather(2, actions).squeeze()

        with torch.no_grad():
            # Next obss goes from [batch-size x hist-len x obs-dim] to
            # [batch-size x hist-len x n-actions] and then goes through gather and squeeze
            # to become [batch-size x hist-len]
            if self.history:
                # print(self.policy_network(next_obss, next_actions, bag_obss, bag_actions).shape)
                argmax = torch.argmax(self.policy_network(next_obss, next_actions, bag_obss, bag_actions),
                                      dim=2).unsqueeze(-1)
                next_obs_q_values = self.target_network(
                    next_obss, next_actions, bag_obss, bag_actions
                )
                next_obs_q_values = next_obs_q_values.gather(2, argmax).squeeze()
                # next_obs_q_values = next_obs_q_values.gather(2, argmax).squeeze(2)
            # here goes BELLMAN
            targets = rewards.squeeze() + (1 - dones.squeeze()) * (
                next_obs_q_values * self.gamma
            )
        #print(q_values)
        # 这是原来的
        #q_values = q_values[:, -self.history :]
        #targets = targets[:, -self.history :]
        # Calculate loss
        loss = F.mse_loss(q_values, targets)
        # Log Losses
        self.qvalue_max.add(q_values.max().item())
        self.qvalue_mean.add(q_values.mean().item())
        self.qvalue_min.add(q_values.min().item())

        self.target_max.add(targets.max().item())
        self.target_mean.add(targets.mean().item())
        self.target_min.add(targets.min().item())

        self.td_errors.add(loss.item())
        # Optimization step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.grad_norm_clip,
            error_if_nonfinite=True,
        )
        # Logging
        self.grad_norms.add(norm.item())

        self.optimizer.step()
        self.num_train_steps += 1

        # if self.num_train_steps % self.target_update_frequency == 0:
        #     self.target_update()
        self.target_update()
        # print("网络更新ing")

    def save_model(self, checkpoint_dir: str, args ) -> None:
        """Save the state of policy network to a file."""
        torch.save(
            {"policy_net_state_dict": self.policy_network.state_dict()},
            checkpoint_dir +args.dataset + ".pt"
        )

    @staticmethod
    def load_model(checkpoint_dir: str, args) -> dict:
        print("正在加载模型")
        """Load the state of policy network from a file."""
        checkpoint = torch.load(checkpoint_dir + args.dataset + ".pt")
        return checkpoint["policy_net_state_dict"]

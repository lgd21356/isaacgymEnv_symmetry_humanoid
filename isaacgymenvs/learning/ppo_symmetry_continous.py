from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch


class A2CAgent(a2c_common.ContinuousA2CBase):

    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound')  # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08,
                                    weight_decay=self.weight_decay)

        self.calc_gradients_times = 0
        self.symmetry_index = self.config.get('symmetry_index', 0.0)

        if self.has_central_value:
            cv_config = {
                'state_shape': self.state_shape,
                'value_size': self.value_size,
                'ppo_device': self.ppo_device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_length': self.seq_length,
                'normalize_value': self.normalize_value,
                'network': self.central_value_config['network'],
                'config': self.central_value_config,
                'writter': self.writer,
                'max_epochs': self.max_epochs,
                'multi_gpu': self.multi_gpu,
                'zero_rnn_on_done': self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn,
                                           self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']


            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo,
                                          curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch,
                                                   self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            if self.calc_gradients_times > 0:
                state_left_side_index = [25, 26, 27, 28, 29, 30, 31, 36, 37, 38, 39, 53, 54, 55, 56, 57, 58, 59, 64, 65,
                                         66, 67, 81, 82, 83, 84, 85, 86, 87, 92, 93, 94, 95, 102, 103, 104, 105, 106,
                                         107, 121, 122, 123, 124, 125, 126, 127, 132, 133, 134, 135]
                state_right_side_index = [18, 19, 20, 21, 22, 23, 24, 32, 33, 34, 35, 46, 47, 48, 49, 50, 51, 52, 60,
                                          61, 62, 63, 74, 75, 76, 77, 78, 79, 80, 88, 89, 90, 91, 96, 97, 98, 99, 100,
                                          101, 114, 115, 116, 117, 118, 119, 120, 128, 129, 130, 131]
                state_common_index = [0, 1, 3, 5, 10, 11, 13, 16, 41, 44, 69, 72, 109, 112]
                state_opposite_index = [2, 4, 6, 7, 8, 9, 12, 14, 15, 17, 40, 42, 43, 45, 68, 70, 71, 73, 108, 110, 111,
                                        113]

                mirror_obs_batch = obs_batch.clone()
                mirror_obs_batch[:, state_opposite_index] = obs_batch.clone()[:, state_opposite_index] * (-1)
                mirror_obs_batch[:, state_left_side_index] = obs_batch.clone()[:, state_right_side_index]
                mirror_obs_batch[:, state_right_side_index] = obs_batch.clone()[:, state_left_side_index]

                mirror_batch_dict = {
                    'is_train': True,
                    'prev_actions': actions_batch,
                    'obs': mirror_obs_batch,
                }

                mirror_res_dict = self.model(mirror_batch_dict)
                mirror_mu_batch = mirror_res_dict['mus']
                action_common_index = [1, 4]
                action_left_side_index = [13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27]
                action_right_side_index = [6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 23]
                action_oppsite_index = [0, 2, 3, 5]

                mirror_mu_batch_clone = mirror_mu_batch.clone()
                mirror_mu_batch[:, action_oppsite_index] = mirror_mu_batch_clone[:, action_oppsite_index] * (-1)
                mirror_mu_batch[:, action_right_side_index] = mirror_mu_batch_clone[:, action_left_side_index]
                mirror_mu_batch[:, action_left_side_index] = mirror_mu_batch_clone[:, action_right_side_index]

                mirror_loss = (mu - mirror_mu_batch).pow(2).mean()
            else:
                mirror_loss = torch.zeros(1, device=self.ppo_device)

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, mirror_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, mirror_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3], losses[4]

            loss = a_loss + 0.5 * c_loss * self.critic_coef + self.symmetry_index * mirror_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            self.calc_gradients_times += 1


            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.diagnostics.mini_batch(self,
                                    {
                                        'values': value_preds_batch,
                                        'returns': return_batch,
                                        'new_neglogp': action_log_probs,
                                        'old_neglogp': old_action_log_probs_batch,
                                        'masks': rnn_masks
                                    }, curr_e_clip, 0)

        self.train_result = (a_loss, c_loss, entropy, \
                             kl_dist, self.last_lr, lr_mul, \
                             mu.detach(), sigma.detach(), b_loss)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu * mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss



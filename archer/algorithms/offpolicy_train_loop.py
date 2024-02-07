from archer.environment import batch_interact_environment
from archer.data import DummyDataset,  ReplayBuffer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from archer.algorithms.archer import ArcherTrainer
import wandb
import threading
import os
import torch
def offpolicy_train_loop(env,\
                eval_env,\
                agent,\
                tokenizer,\
                accelerator,\
                warmup_iter: int = 20,
                rollout_size: int = 50,\
                eval_size: int = 1,
                batch_size: int = 2,
                capacity: int = 500000,
                iterations: int = 10,\
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                env_idx:int = None,\
                do_sample: bool = False,\
                temperature: float = 2.0,\
                critic_lr: float= 1e-3,\
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                tau: float = 0.1,
                use_wandb: bool = False,
                env_load_path: str = '',
                actor_epochs: int = 3,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 25,
                eval_freq: int = 25,
                **kwargs):
    trainer = ArcherTrainer(agent=agent,\
                         accelerator=accelerator,\
                            tokenizer=tokenizer,\
                            critic_lr = critic_lr,\
                            lm_lr = lm_lr,\
                            gamma = gamma,\
                            tau = tau,\
                            epochs = epochs,\
                            actor_epochs = actor_epochs,
                            grad_accum_steps=grad_accum_steps,
                            max_grad_norm=max_grad_norm)
    replay_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    if save_path is not None and accelerator.is_main_process:
        if os.path.exists(os.path.join(save_path, 'trainer.pt')):
            # print("Not using existing checkpoint")
            print("Loading from checkpoint")
            trainer.load(os.path.join(save_path, 'trainer.pt'))
            replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))
        else:
            print("Creating new checkpoint directory")
            os.makedirs(save_path, exist_ok=True)
    agent.prepare()
    #main training loop
    print(">>>start iterations")
    for i in tqdm(range(iterations)):
        # print(">>>Interacting with Environment")
        trajectories = batch_interact_environment(agent = agent,\
                                        tokenizer= tokenizer,\
                                        env = env,\
                                        num_trajectories= rollout_size,\
                                        env_idx = env_idx,
                                        use_tqdm=False,
                                        temperature=temperature,
                                        do_sample=do_sample)
        info = {"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in trajectories]),\
                "rollout.max": np.max([d[0]["trajectory_reward"] for d in trajectories]),\
                "rollout.min": np.min([d[0]["trajectory_reward"] for d in trajectories])}
        if (i+1) % eval_freq == 0:
            old_sample = agent.do_sample
            agent.do_sample = False
            eval_trajectories =  batch_interact_environment(agent = agent,\
                                                tokenizer= tokenizer,\
                                                env = eval_env,\
                                                num_trajectories=  max(eval_size, eval_env.bsize),\
                                                env_idx = env_idx,
                                                use_tqdm=False,
                                                temperature=temperature,
                                                do_sample=do_sample)
            agent.do_sample = old_sample
            info.update({"eval_rollout.mean": np.mean([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                    "eval_rollout.max": np.max([d[0]["trajectory_reward"] for d in eval_trajectories]),\
                    "eval_rollout.min": np.min([d[0]["trajectory_reward"] for d in eval_trajectories]),})
        data = sum(trajectories, [])
        for t in data:
            replay_buffer.insert(**t)
        info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),\
                "rollout.reward.max": np.max([d["reward"] for d in data]),\
                "rollout.reward.min": np.min([d["reward"] for d in data])})
        # data = list(filter(lambda x: x["reward"] >0, data))
        print("Training")
        info.update(trainer.update(replay_buffer, no_update_actor = (i < warmup_iter)))
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'))
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
    # return model
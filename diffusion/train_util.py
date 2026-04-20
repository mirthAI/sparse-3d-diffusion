import copy
import functools
import os
import numpy as np
import time

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import SimpleITK as sitk

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        args,
        model,
        diffusion,
        data,
        schedule_sampler
    ):
        self.diffusion = diffusion
        self.data = data
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.ema_rate = (
            [args.ema_rate]
            if isinstance(args.ema_rate, float)
            else [float(x) for x in args.ema_rate.split(",")]
        )
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.result_folder = args.result_folder
        self.schedule_sampler = schedule_sampler
        self.weight_decay = args.weight_decay
        self.type = args.type
        self.f_steps = args.f_steps
        self.dwt = args.dwt

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.gpu_id)

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=False
        )

        self.opt = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        self.ddp_model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_and_sync_parameters(self): # loading a copy of the trained model weights for commencing training from where it stopped earlier
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            #if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=f"cuda:{self.gpu_id}"
                )
            )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate): # loading a copy of the exponential moving average (ema) checkpoints from where it stopped earlier
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            #if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=f"cuda:{self.gpu_id}"
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self): # loading a copy ofthe trained optimizer
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=f"cuda:{self.gpu_id}"
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        if self.dwt:
            self._loss_fn = lambda full, low, t: self.diffusion.training_losses_dwt(self.ddp_model, full, low, t)
        else:
            self._loss_fn = lambda full, low, t: self.diffusion.training_losses(self.ddp_model, full, low, t)

        if self.type == "fast":
            self._sample_fn = lambda bsz: self.schedule_sampler.sample_fast(bsz, self.f_steps, self.gpu_id)
        elif self.type == "ddpm":
            self._sample_fn = lambda bsz: self.schedule_sampler.sample(bsz, self.gpu_id)
        else:
            raise ValueError(f"Unknown type: {self.type}")

        accu_time=0.0
        time_start = time.perf_counter()

        while True:
            batch = next(self.data)
            self.run_step(batch)

            iter_time = time.perf_counter() - time_start
            accu_time += iter_time
            time_start = time.perf_counter()

            if self.step % self.log_interval == 0:
                avg_time = accu_time / self.log_interval
                print(f"[RANK{self.global_rank}: local GPU{self.gpu_id}] time{avg_time:.2f} seconds")
                logger.dumpkvs()
                accu_time = 0.0

            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()

            self.step += 1


    def run_step(self, batch):
        self.mp_trainer.zero_grad()

        full_batch = batch['full_res'].unsqueeze(1).to(self.gpu_id)
        low_batch = batch['low_res'].unsqueeze(1).to(self.gpu_id)

        t, weights = self._sample_fn(full_batch.shape[0])

        losses = self._loss_fn(full_batch, low_batch, t)

        loss = (losses["loss"]).mean()
        # loss = (losses["loss"] * weights).mean()
        
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        self.mp_trainer.backward(loss)

        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self.log_step()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def log_step(self): # logger for dispalying current status of training
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self): # saving diffusion model 
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"

                with bf.BlobFile(bf.join(self.result_folder, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier(device_ids=[self.gpu_id])


def parse_resume_step_from_filename(filename): # parsing model checkpoint files
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir(): # save directory location 
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()
    # return '/blue/weishao/vi.sivaraman/Tasks/Image_Reconstruction/3D_Data/ddpm_results'


def find_resume_checkpoint(): # finding last resume checkpoint 
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate): # finding latest ema checkpoint automatically 
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses): # logs the losses
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quartiles (four quartiles, in particular). # It computes the quartile for each timestep based on its position relative to the total number of timesteps in the diffusion process. This essentially divides the training process into four equal parts (quartiles)
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss) # logs the loss value for each quartile 

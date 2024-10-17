import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch as th
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_classifier_and_diffusion


device = torch.device("cuda")

noised=True

iterations=3

anneal_lr=True

# dict_keys(['image_size', 'classifier_use_fp16', 'classifier_width', 
#            'classifier_depth', 'classifier_attention_resolutions', 
#            'classifier_use_scale_shift_norm', 'classifier_resblock_updown', 
#            'classifier_pool', 'learn_sigma', 'diffusion_steps', 'noise_schedule', 
#            'timestep_respacing', 'use_kl', 'predict_xstart', 
#            'rescale_timesteps', 'rescale_learned_sigmas'])
model, diffusion = create_classifier_and_diffusion(
            image_size = 64, classifier_use_fp16=False, classifier_width=128, 
            classifier_depth=2, classifier_attention_resolutions='32,16,8', 
            classifier_use_scale_shift_norm=True, classifier_resblock_updown=True, 
            classifier_pool='attention', learn_sigma=False, diffusion_steps=1000, noise_schedule='linear', 
            timestep_respacing='', use_kl=False, predict_xstart=False, 
            rescale_timesteps=False, rescale_learned_sigmas=False
            )   

model = model.to(device)

resume_step = 0

if noised:
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False, initial_lg_loss_scale=16.0)


data = load_data(
        data_dir='../data/imagenet_64/train',
        batch_size=256,
        image_size=64,
        class_cond=True,
        random_crop=True,
    )

val_data = None

opt = AdamW(mp_trainer.master_params, lr=0.0003, weight_decay=0.05)

print("training classifier model...")

def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(device)

        batch = batch.to(device)
        # Noisy images
        if noised:
            t, _ = schedule_sampler.sample(batch.shape[0], device)
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=device)

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(-1, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )

            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):

   
    th.save(
        mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
       f"./models/guide_diffussion/model{step:06d}.pt",
    )
    th.save(opt.state_dict(), f"./models/guide_diffussion/opt{step:06d}.pt")


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

for step in range(iterations - resume_step):
    
    if anneal_lr:
        set_annealed_lr(opt, 0.0003, (step + resume_step) / iterations)
    forward_backward_log(data)
    mp_trainer.optimize(opt)
    
    print("iter is : ",step)

save_model(mp_trainer, opt, step + resume_step)


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import torch.nn.functional as F
from guided_diffusion.unet import UNetModel, EncoderUNetModel
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps

device = torch.device("cuda")

image_size = 64

class_cond = True

classifier_scale=1.0

clip_denoised=True

classifier_use_fp16=False

use_ddim=False

batch_size = 4

num_samples = 8

NUM_CLASSES = 1000


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )





    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    print("attention_ds = [] is : ",attention_ds)

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )

print("扩散模型创建中。。。")
#  dict_keys(['image_size', 'num_channels', 'num_res_blocks', 'num_heads', 
#             'num_heads_upsample', 'num_head_channels', 'attention_resolutions', 
#             'channel_mult', 'dropout', 'class_cond', 'use_checkpoint', 'use_scale_shift_norm', 
#             'resblock_updown', 'use_fp16', 'use_new_attention_order', 'learn_sigma', 
#             'diffusion_steps', 'noise_schedule', 'timestep_respacing', 'use_kl', 'predict_xstart', 
#             'rescale_timesteps', 'rescale_learned_sigmas'])
model, diffusion = create_model_and_diffusion(
                    image_size=64, num_channels=192, num_res_blocks=3, num_heads=4,
                    num_heads_upsample=-1, num_head_channels=64, attention_resolutions='32,16,8',
                    channel_mult='', dropout=0.1, class_cond=True, use_checkpoint=False, 
                    use_scale_shift_norm=True,resblock_updown=True, use_fp16=False, 
                    use_new_attention_order=True, learn_sigma=True,
                    diffusion_steps=1000, noise_schedule='cosine', timestep_respacing='250', 
                    use_kl=False, predict_xstart=False, rescale_timesteps=False, rescale_learned_sigmas=False)


model.load_state_dict(torch.load('models/guide_diffussion/64x64_diffusion.pt', map_location="cpu", weights_only=True))
model = model.to(device)
model.eval()

print("扩散模型加载成功！！！")

print("引导模型创建中。。。")
# dict_keys(['image_size', 'classifier_use_fp16', 'classifier_width', 
#            'classifier_depth', 'classifier_attention_resolutions', 
#            'classifier_use_scale_shift_norm', 'classifier_resblock_updown', 
#            'classifier_pool']
classifier = create_classifier(
                            image_size=64, classifier_use_fp16=False, classifier_width=128, 
                            classifier_depth=4, classifier_attention_resolutions='32,16,8', 
                            classifier_use_scale_shift_norm=True, classifier_resblock_updown=True, 
                            classifier_pool='attention')
classifier.load_state_dict(torch.load('models/guide_diffussion/64x64_classifier.pt', map_location="cpu", weights_only=True))
classifier = classifier.to(device)
classifier.eval()

if classifier_use_fp16:
    classifier.convert_to_fp16()
classifier.eval()

print("引导模型成创建成功！！！ ")


def cond_fn(x, t, y=None):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]

        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

def model_fn(x, t, y=None):
    assert y is not None
    return model(x, t, y if class_cond else None)

print("采样中。。。 ")
all_images = []
all_labels = []
while len(all_images) * batch_size < num_samples:

    model_kwargs = {}
    # 随机生成类别
    classes = torch.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=device)

    model_kwargs["y"] = classes
    #定义采样方式
    sample_fn = (diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop)

    #开始采样
    sample = sample_fn(
        model_fn,
        (batch_size, 3, image_size, image_size),
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=device,
    )
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    samples = sample.contiguous()

    all_images.append(samples.cpu().numpy())
    all_labels.append(classes.cpu().numpy())


arr = np.concatenate(all_images, axis=0)
arr = arr[: num_samples]
#print("arr is : ",arr.shape)

label_arr = np.concatenate(all_labels, axis=0)
label_arr = label_arr[:num_samples]
# print("label_arr is : ",label_arr)
# print("label_arr is : ",label_arr.shape)

#获取 arr 的形状信息，将每个维度的大小转换为字符串，并用 "x" 连接起来。例如，如果 arr.shape 是 (100, 64, 64, 3)，
#则 shape_str 会被设置为 "100x64x64x3"。这个信息用于文件命名，便于查看保存文件中的样本形状。
shape_str = "x".join([str(x) for x in arr.shape])
out_path = os.path.join(f"./samples_{shape_str}.npz")
np.savez(out_path, arr, label_arr)

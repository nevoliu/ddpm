from Diffusion.Inference import infer


def main(model_config = None):
    modelConfig = {
        "state": "eval", # train or eval
        "epoch": 200,
        "batch_size": 100,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", 
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "../data/ffhq/compare/ddim/",
        "nrow": 1
        }
    infer(modelConfig)


if __name__ == '__main__':
    main()

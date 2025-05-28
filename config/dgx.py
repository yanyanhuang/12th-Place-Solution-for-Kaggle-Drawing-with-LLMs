import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.num_epochs = 100
    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def svg_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/svg_e4000")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 30
    config.sample.guidance_scale=4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 8
    config.sample.test_batch_size = 12

    config.train.lora_path = 'jieliu/SD3.5M-FlowGRPO-PickScore'
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.0004
    # kl reward
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/sd35M'
    config.reward_fn = {
        "svg": 1.0,
    }
    
    config.prompt_fn = "svg"

    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()

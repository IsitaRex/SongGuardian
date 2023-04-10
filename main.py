import wandb
import torch
import torchaudio
import argparse
import src.utils as utils

# parameters
DEVICE = "cpu"
# if torch.cuda.is_available():
#     DEVICE = "cuda"
# elif torch.backends.mps.is_available():
#     DEVICE = "mps"

SEED = 121212


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main(config: dict):
    wandb.init(project=config["task"], config=config)
    config["sample_rate"] = 44100
    if config["task"] == "create_artifact":
        utils.create_artifact(config)
    elif config["task"] == "RNN":
        utils.task_rnn(config)
    elif config["task"] == "CNN":
        utils.task_cnn(config)
    else:
        raise ValueError("Task not found.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--task", type=str, default="RNN")
    parser.add_argument("--seed", type=int, default=121212)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--use_wandb_offline", type=bool, default=False)
    # parser.add_argument("--device", type=str, default=DEVICE)
    config = vars(parser.parse_args())
    config["device"] = DEVICE
    utils.setup_wandb_logging(config)
    main(config)
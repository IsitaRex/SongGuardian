import wandb
import torch
import torchaudio
import argparse
import src.utils as utils
import src.data_utils as data_utils

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
    if config["task"] == "build_dataset":
        data_utils.build_dataset("data", config["features"])
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--task", type=str, default="RNN")
    parser.add_argument("--seed", type=int, default=121212)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--use_wandb_offline", type=bool, default=False)
    parser.add_argument("--features", type=str, nargs='+', help="List of features to extract")


    # parser.add_argument("--device", type=str, default=DEVICE)
    config = vars(parser.parse_args())
    config["device"] = DEVICE
    utils.setup_wandb_logging(config)
    # Ensure that features are specified when task is "build_dataset"
    if config["task"] == "build_dataset" and config["features"] is None:
        raise ValueError("Error: --features must be specified when task is 'build_dataset'.")
    main(config)
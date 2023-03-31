import wandb
import os


def setup_wandb_logging(config):

    # Configure WandB settings
    if not config["use_wandb"]:
        wandb_settings = wandb.Settings(
            mode="disabled",
            program=__name__,
            program_relpath=__name__,
            disable_code=True,
        )
        wandb.setup(wandb_settings)
    else:
        if config["use_wandb_offline"]:
            os.environ["WANDB_MODE"] = "offline"
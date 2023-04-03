import wandb
import torch
import src.utils as utils
import src.models as models
import src.data_utils as data_utils

SEED = 121212


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def create_artifact(config):
    # create an artifact for the data
    data_artifact = wandb.Artifact('data', type='dataset')
    # add the mp3 files to the artifact and the labels
    data_artifact.add_dir('data_cut/Like', name='data/Like')
    data_artifact.add_dir('data_cut/Dislike', name='data/Dislike')
    # log the artifact to wandb
    wandb.log_artifact(data_artifact)


def task_rnn(config):
    path = "data_cut/"
    train_loader, test_loader = data_utils.get_dataloaders(path,config)
    model = models.RNN(56, 3, 3, num_classes = 2, device = config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    model, train_losses, test_losses, test_accs = utils.training_loop(model, train_loader, test_loader, optimizer, criterion, config['epochs'], config['device'])
    return model, train_losses, test_losses, test_accs

def task_cnn(config):
    path = "data_cut/"
    train_loader, test_loader = data_utils.get_dataloaders(path, config)
    model = models.CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    model, train_losses, test_losses, test_accs = utils.training_loop(model, train_loader, test_loader, optimizer, criterion, config['epochs'], config['device'])
    return model, train_losses, test_losses, test_accs
import torch

import src.utils as utils
import src.models as models
import src.data_utils as data_utils
SEED = 121212


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def task_rnn(config):
    '''
    Task to train a RNN model.
    '''
    if config["features"] is None:
        raise ValueError("Error: --features must be specified when task is 'task_rnn'.")
    path = "data/"
    train_loader, test_loader = data_utils.get_dataloaders(path,config)
    # get input size
    input_size = train_loader.dataset[0][0].shape[1]
    model = models.RNN(input_size, 3, 3, num_classes = 2, device = config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    model, train_losses, test_losses, test_accs = utils.training_loop(model, train_loader, test_loader, optimizer, criterion, config['epochs'], config['device'])
    # save the model in model directory
    torch.save(model.state_dict(), "models/rnn_0.0005.pt")
    return model, train_losses, test_losses, test_accs
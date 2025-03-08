import torch
import wandb
# import f1 score
from sklearn.metrics import f1_score, precision_score, recall_score


def train(model, train_loader, optimizer, criterion, epoch, device):
  '''
  Train the model.

  Args:
    model: model to train
    train_loader: train dataloader
    optimizer: optimizer
    criterion: loss function
    epoch: epoch number
    device: device to use

  Returns:
    model: trained model
    train_loss: train loss
  '''
  model.train()
  train_loss = 0
  ii  = 0
  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      output = model(data)
      output = output.float()
      output.requires_grad_(True)
      
      # loss = criterion(output.to(torch.float32), target.detach().to(torch.float32))
      loss = criterion(output, target)
      optimizer.zero_grad()
      loss.backward()
      
      optimizer.step()
      train_loss += loss.item()
      if batch_idx % 100 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))
  train_loss /= len(train_loader.dataset)

  return model, train_loss
 

def test(model, test_loader, criterion, device, confusion_matrix = False):
  '''
  Test the model.
  
  Args:
    model: model to test
    test_loader: test dataloader
    criterion: loss function
    device: device to use
  Returns:
    model: model
    test_loss: loss
    correct: number of correct predictions
    f1: f1 score
    precision: precision score
    recall: recall score
  '''

  model.eval()
  test_loss = 0
  correct = 0
  TP, FP, TN, FN = 0, 0, 0, 0

  if confusion_matrix:
     ground_truth = []
     predictions = []
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          output = output.float()
          test_loss += criterion(output, target).item()  # sum up batch loss
          correct += (output.argmax(dim=-1) == target).sum()

          # compute batch true positives, false positives, true negatives, false negatives
          TP += ((output.argmax(dim=-1) == 1) & (target == 1)).sum()
          FP += ((output.argmax(dim=-1) == 1) & (target == 0)).sum()
          TN += ((output.argmax(dim=-1) == 0) & (target == 0)).sum()
          FN += ((output.argmax(dim=-1) == 0) & (target == 1)).sum()

          if confusion_matrix:
            ground_truth.extend(target.tolist())
            predictions.extend(output.argmax(dim=-1).tolist())
          
  test_loss /= len(test_loader.dataset)
  

  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  f1 = 2 * precision * recall / (precision + recall)

  
  if confusion_matrix:
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=ground_truth, preds=predictions,
                        class_names=['like', 'dislike'])})

  return model, test_loss, correct, f1, precision, recall

def training_loop(model, train_loader, test_loader, optimizer, criterion, epochs, device):
  '''
  Training loop for the model.

  Args:
    model: model to train
    train_loader: train dataloader
    test_loader: test dataloader
    optimizer: optimizer to use
    criterion: loss function
    epochs: number of epochs to train for
    device: device to train on
  
  Returns:
    model: trained model
    train_losses: list of train losses
    test_losses: list of test losses
    test_accs: list of test accuracies
  '''
  train_losses = []
  test_losses = []
  test_accs = []
  for epoch in range(1, epochs + 1):
      model, train_loss = train(model, train_loader, optimizer, criterion, epoch, device)
      model, test_loss, correct,f1, precision, recall = test(model, test_loader, criterion, device)
      train_losses.append(train_loss)
      test_losses.append(test_loss)
      test_accs.append(correct/len(test_loader.dataset))
      wandb.log({"train_loss": train_loss, "test_loss": test_loss, "test_acc": correct/len(test_loader.dataset)})
      wandb.log({"f1": f1, "precision": precision, "recall": recall})

  # Do a final test run and log the confusion matrix
  test(model, test_loader, criterion, device, confusion_matrix=True)

  return model, train_losses, test_losses, test_accs

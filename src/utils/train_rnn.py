import torch
import wandb
# import f1 score
from sklearn.metrics import f1_score, precision_score, recall_score


def train(model, train_loader, optimizer, criterion, epoch, device):
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
 

def test(model, test_loader, criterion, device):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          output = output.float()
          test_loss += criterion(output, target).item()  # sum up batch loss
          #pred = output  # get the index of the max log-probability
          #correct += pred.eq(target.view_as(pred)).sum().item()
          correct += (output.argmax(dim=-1) == target).sum()
  # get f1 score, precision, recall
  
  f1 = f1_score(target, output.argmax(dim=-1), average='macro')
  precision = precision_score(target, output.argmax(dim=-1), average='macro')
  recall = recall_score(target, output.argmax(dim=-1), average='macro')
  test_loss /= len(test_loader.dataset)
  
  return model, test_loss, correct, f1, precision, recall

def training_loop(model, train_loader, test_loader, optimizer, criterion, epochs, device):
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

  return model, train_losses, test_losses, test_accs

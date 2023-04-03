import os
import torchaudio
import torch
import pydub
from torch.utils.data import Dataset

class SongsDataset(Dataset):

    def __init__(self, path,  sample_rate=44100, device = 'cpu'):
        self.path = path
        self.like = self._count_files_in_directory(self.path + '/Like')
        self.dislike = self._count_files_in_directory(self.path + '/Dislike')
        self.device = device
        self.sample_rate = sample_rate
        self.use_mfcc = True

    def __len__(self):
        return self.like + self.dislike

    def __getitem__(self, idx):
        song_sample, sr, label = self._get_sample_file(idx)
        song_sample = self._mix_down(song_sample)
        if self.use_mfcc:
          song_sample = self._get_mfcc_features(song_sample, sr)
        else:
          mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
          )
          song_sample = mel_spectrogram(song_sample)
        return song_sample, label

    def _get_mfcc_features(self, signal, sample_rate):
      '''
      This function takes the raw mfcc features and returns the mean and standard deviation
      '''
      mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=14)(signal)
      mean = torch.mean(mfcc, dim=2)
      std = torch.std(mfcc, dim=2)
      mx = torch.max(mfcc, dim=2)[0]
      mn = torch.min(mfcc, dim=2)[0]
      return torch.cat((mean, std, mx, mn), dim=1)

    
    def _count_files_in_directory(self, path):
      count = 0
      for p in os.scandir(path):
          if p.is_file():
              count += 1
      
      return count

    def _get_sample_file(self, index):

        if index < self.like:
          label = 'Like'
        else:
          index %= 100
          label = 'Dislike'
        
        # if index has less than three digits, add zeros to the left
        index = str(index+1)
        while(len(index)<3):
            index = '0'+index
        path = os.path.join(self.path, label, str(index)+'.wav')
        signal, sample_rate = torchaudio.load(path)
        signal = signal.to(self.device)
        # assign 0 to like and 1 to dislike
        if label == 'Like':
          label = 0
        else:
          label = 1
        return signal, sample_rate, label

    def _convert_to_wav(self, path):
      for p in os.scandir(path):
          if p.is_file():
              print(p.path)
              sound = pydub.AudioSegment.from_mp3(p.path)
              sound.export(p.path[:-3]+'wav', format="wav")
              os.remove(p.path)

    def _mix_down(self, song_sample):
        if song_sample.shape[0] > 1:
            song_sample = torch.mean(song_sample, 0, True)
        return song_sample

def get_dataloaders(path, config):
    
    batch_size = config['batch_size']
    sample_rate = config['sample_rate']
    device = config['device']
    dataset = SongsDataset(path, sample_rate=sample_rate, device=device)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

if __name__ == '__main__':
  PATH = 'data_cut'
  SAMPLE_RATE = 44100
  # parameters
  DEVICE = "cpu"
#   if torch.cuda.is_available():
#       DEVICE = "cuda"
#   elif torch.backends.mps.is_available():
#       DEVICE = "mps"

  mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE)
  mfcc = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE)
  dataset = SongsDataset('data_cut', sample_rate= SAMPLE_RATE, device= DEVICE)
  # print(dataset[0][1])
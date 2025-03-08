import os
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset
import pydub
import tqdm
import numpy as np
class SongsDataset(Dataset):
    def __init__(self, path, features=["timbre"], sample_rate=44100, device='cpu'):
        self.path = path
        self.like = self._count_files_in_directory(os.path.join(self.path, 'Like'))
        self.dislike = self._count_files_in_directory(os.path.join(self.path, 'Dislike'))
        self.device = device
        self.sample_rate = sample_rate
        self.features = features

    def __len__(self):
        return self.like + self.dislike

    def __getitem__(self, idx):
        song_sample, sr, label = self._get_sample_file(idx)
        song_sample = self._mix_down(song_sample)
        
        feature_data = []
        feature_extractors = {
            "timbre": self._get_timbre_features,
            "harmony": self._get_harmony_features,
            "rhythm": self._get_rhythm_features,
            "dynamics": self._get_dynamics_features,
        }

        for feature in self.features:
            if feature in feature_extractors:
                feature_data.append(feature_extractors[feature](song_sample, sr))
            else:
                raise ValueError(f"Unsupported feature: {feature}. Available features: {list(feature_extractors.keys())}")

        # Concatenate all features along the feature dimension
        song_features = torch.cat(feature_data, dim=1) if feature_data else torch.tensor([])

        return song_features, label

    def _get_timbre_features(self, signal, sample_rate):
        """Extract MFCC-based timbre features (mean, std, max, min)."""
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, 
            n_mfcc=14,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
        )(signal)
        mean = torch.mean(mfcc, dim=2)
        std = torch.std(mfcc, dim=2)
        mx = torch.max(mfcc, dim=2)[0]
        mn = torch.min(mfcc, dim=2)[0]
        return torch.cat((mean, std, mx, mn), dim=1)

    def _get_harmony_features(self, signal, sample_rate):
        """Extract Chroma features for harmony representation."""
        chroma_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=12  # Chroma-like features
        )
        chroma = chroma_transform(signal)
        return torch.cat((chroma.mean(dim=2), chroma.var(dim=2)), dim=1)
    
    import librosa

    def _get_tempo(self, signal, sample_rate):
        """Extracts tempo (BPM) using librosa's beat tracking."""
        # Ensure the signal is mono and convert it to a NumPy array
        if signal.ndim > 1:  # If multi-channel, mix down to mono
            signal = torch.mean(signal, dim=0)  # Mix down to mono
        signal = signal.cpu().numpy()  # Convert to NumPy array

        # Compute the onset envelope
        onset_env = librosa.onset.onset_strength(y=signal, sr=sample_rate)

        # Extract tempo
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)

        # Ensure tempo is a scalar value
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]  # Take the first value if it's an array

        # Return as a tensor in (batch, 1) format
        return torch.tensor([[tempo]], device=self.device)

    
    def _get_spectral_flux(self, signal, sample_rate):
        """Measures how the spectrum changes over time (rhythmic complexity)."""
        spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(signal)
        flux = torch.norm(spectrogram[:, :, 1:] - spectrogram[:, :, :-1], dim=1).mean(dim=1, keepdim=True)
        return flux

    def _get_pulse_clarity(self, signal, sample_rate):
        """Estimates pulse clarity using onset strength autocorrelation."""
        onset_env = librosa.onset.onset_strength(y=signal.cpu().numpy(), sr=sample_rate)
        autocorr = librosa.autocorrelate(onset_env)
        clarity = torch.tensor([[autocorr.max()]], device=self.device)  # Normalize range
        return clarity

    def _get_rhythm_features(self, signal, sample_rate):
        """Extract various rhythm-related features."""
        zero_crossings = torch.mean((torch.diff(torch.sign(signal)) != 0).float(), dim=1, keepdim=True)
        tempo = self._get_tempo(signal, sample_rate)
        spectral_flux = self._get_spectral_flux(signal, sample_rate)
        pulse_clarity = self._get_pulse_clarity(signal, sample_rate)

        return torch.cat((zero_crossings,tempo,  spectral_flux, pulse_clarity), dim=1)


    def _get_dynamics_features(self, signal, sample_rate):
        """Extract dynamics features using RMS energy."""
        rms = torchaudio.transforms.AmplitudeToDB()(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(signal))
        return rms.mean(dim=2)

    def _count_files_in_directory(self, path):
        return sum(1 for p in os.scandir(path) if p.is_file())

    def _get_sample_file(self, index):
        """Loads a song sample from either Like or Dislike folder based on index."""
        label = 'Like' if index < self.like else 'Dislike'
        index = str((index % self.like) + 1).zfill(3)
        path = os.path.join(self.path, label, f"{index}.wav")
        signal, sample_rate = torchaudio.load(path)
        signal = signal.to(self.device)
        label = 0 if label == 'Like' else 1
        return signal, sample_rate, label

    def _mix_down(self, song_sample):
        """Convert stereo to mono by averaging channels."""
        return torch.mean(song_sample, dim=0, keepdim=True) if song_sample.shape[0] > 1 else song_sample

    def _convert_to_wav(self, path):
        """Convert MP3 files to WAV format."""
        for p in os.scandir(path):
            if p.is_file() and p.path.endswith(".mp3"):
                sound = pydub.AudioSegment.from_mp3(p.path)
                sound.export(p.path[:-4] + '.wav', format="wav")
                os.remove(p.path)


def build_dataset(path, features=["timbre"], sample_rate=44100, device='cpu'):
    '''Builds dataset and saves it to distk'''
    dataset = SongsDataset(path=path, features=features, sample_rate=sample_rate, device=device)
    print(f"Building dataset with features: {features}")
    print(len(dataset))
    features_list = []
    labels_list = []
    for i,(sample, label) in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        features_list.append(sample)
        labels_list.append(label)
        if i > 200: # TODO: understand why this is necessary
            break
    
    print("Saving dataset to disk...")
    features_tensor = torch.stack(features_list)
    labels_tensor = torch.tensor(labels_list)
    suffix = "_".join(features)
    torch.save(features_tensor, f"{path}/Features/{suffix}.pt")
    torch.save(labels_tensor, f"{path}/Features/labels.pt")

def get_dataloaders(path, config):
    batch_size = config['batch_size']
    sample_rate = config['sample_rate']
    device = config['device']
    dataset = SongsDataset(path=path, features=config["features"], sample_rate=sample_rate, device=device)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

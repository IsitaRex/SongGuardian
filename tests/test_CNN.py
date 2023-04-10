from src import CNN
from torchsummary import summary

def test_CNN():
    cnn = CNN()
    assert cnn is not None
    
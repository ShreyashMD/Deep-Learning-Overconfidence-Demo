
import torchvision
import os

def download_data():
    print("Checking/Downloading CIFAR-10 dataset...")
    try:
        # download=True will check integrity and download if needed
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        print("Dataset download/verification complete!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_data()

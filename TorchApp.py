import torch

if __name__ == "__main__":
    print(torch.__version__)
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
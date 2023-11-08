import argparse
import os

import gdown

transformer_25k = "https://drive.google.com/uc?id=10XRHWwdiLZuuNHEhd0L_eBrKwoAJUqd1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model parser")
    parser.add_argument("checkpoint_name", choices=["transformer"])
    args = parser.parse_args()

    destination_path = f"models/{args.checkpoint_name}_checkpoints/"
    os.makedirs(destination_path, exist_ok=True)
    destination_path += "{args.checkpoint_name}.pth"

    if args.checkpoint_name == "transformer":
        gdown.download(transformer_25k, destination_path, quiet=False)

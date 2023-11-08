import argparse
import os

import gdown

# Checkpoint storage
transformer_25k = "https://drive.google.com/uc?id=10XRHWwdiLZuuNHEhd0L_eBrKwoAJUqd1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model parser")
    parser.add_argument("checkpoint_name", choices=["transformer"])
    args = parser.parse_args()

    # Define the destination path for the downloaded checkpoint
    destination_path = f"models/{args.checkpoint_name}_checkpoints/"
    os.makedirs(destination_path, exist_ok=True)
    destination_path += f"{args.checkpoint_name}.pth"

    if args.checkpoint_name == "transformer":
        # Download the specified model checkpoint
        gdown.download(transformer_25k, destination_path, quiet=False)

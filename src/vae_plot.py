import argparse
import torch
import imageio
import os
import numpy as np
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms
import matplotlib.pyplot as plt

def visualize_and_save_tensor(tensor, output_path, filename, title="Tensor Visualization"):
    """
    Visualizes a tensor as an image and saves it.

    Parameters:
    - tensor (torch.Tensor): The tensor to visualize.
    - output_path (str): The directory to save the plot.
    - filename (str): The filename for the saved plot.
    - title (str): The title of the plot.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tensor = tensor.to(dtype=torch.float32)

    if tensor.ndim == 4:  # Batch of frames
        tensor = tensor[0, :, :, :]
    if tensor.ndim == 3:  # Latent tensor with channels
        # Option 1: Select a single channel
        tensor = tensor[0, :, :]  # Choose the first channel
        # Option 2: Uncomment below to average across channels
        # tensor = tensor.mean(dim=0)

    tensor = tensor.cpu().numpy()
    tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor) + 1e-5)

    plt.imshow(tensor, cmap="viridis")
    plt.title(title)
    plt.axis("off")
    filepath = os.path.join(output_path, f"{filename}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

def encode_video(model_path, video_path, dtype, device, output_path):
    """
    Encodes a video into latent representations using a pre-trained CogVideoX model.
    Saves the first frame and its latent visualization.

    Parameters:
    - model_path (str): Path to the pre-trained model.
    - video_path (str): Path to the video file.
    - dtype (torch.dtype): Data type for computation.
    - device (str): Device for computation.
    - output_path (str): Directory to save visualizations.

    Returns:
    - torch.Tensor: Encoded latent tensor.
    """
    print("Encoding video...")
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.enable_slicing()
    model.enable_tiling()

    video_reader = imageio.get_reader(video_path, "ffmpeg")
    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    video_reader.close()

    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)

    # Save the first frame for visualization
    visualize_and_save_tensor(frames_tensor[0, :, 0], output_path, "original_frame", "First Video Frame")

    with torch.no_grad():
        encoded_frames = model.encode(frames_tensor)[0].sample()

    # Visualize the first latent frame
    visualize_and_save_tensor(encoded_frames[0, :, 0], output_path, "latent_frame", "First Latent Frame")
    print("Encoding completed.")
    return encoded_frames


def decode_video(model_path, encoded_tensor_path, dtype, device, output_path):
    """
    Decodes latent representations into video frames using a pre-trained CogVideoX model.
    Saves the first decoded frame visualization.

    Parameters:
    - model_path (str): Path to the pre-trained model.
    - encoded_tensor_path (str): Path to the encoded tensor file.
    - dtype (torch.dtype): Data type for computation.
    - device (str): Device for computation.
    - output_path (str): Directory to save visualizations.

    Returns:
    - torch.Tensor: Decoded video frames.
    """
    print("Decoding video...")
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)

    with torch.no_grad():
        decoded_frames = model.decode(encoded_frames).sample

    # Save the first decoded frame for visualization
    visualize_and_save_tensor(decoded_frames[0, :, 0], output_path, "decoded_frame", "First Decoded Frame")
    print("Decoding completed.")
    return decoded_frames


def save_video(tensor, output_path):
    """
    Saves video frames to a video file.

    Parameters:
    - tensor (torch.Tensor): Tensor of video frames.
    - output_path (str): Path to save the video file.
    """
    print("Saving video...")
    tensor = tensor.to(dtype=torch.float32)
    frames = tensor[0].permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)

    video_writer = imageio.get_writer(os.path.join(output_path, "output.mp4"), fps=8)
    for frame in frames:
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Video saved at {os.path.join(output_path, 'output.mp4')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVideoX encode/decode demo with visualization")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the CogVideoX model")
    parser.add_argument("--video_path", type=str, help="Path to the video file (for encoding)")
    parser.add_argument("--encoded_path", type=str, help="Path to the encoded tensor file (for decoding)")
    parser.add_argument("--output_path", type=str, default=".", help="Directory to save outputs")
    parser.add_argument(
        "--mode", type=str, choices=["encode", "decode", "both"], required=True, help="Mode: encode, decode, or both"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="Data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for computation (e.g., 'cuda' or 'cpu')"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    os.makedirs(args.output_path, exist_ok=True)

    if args.mode == "encode":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device, args.output_path)
        torch.save(encoded_output, os.path.join(args.output_path, "encoded.pt"))
        print(f"Encoded video saved to {os.path.join(args.output_path, 'encoded.pt')}")

    elif args.mode == "decode":
        assert args.encoded_path, "Encoded tensor path must be provided for decoding."
        decoded_output = decode_video(args.model_path, args.encoded_path, dtype, device, args.output_path)
        save_video(decoded_output, args.output_path)

    elif args.mode == "both":
        assert args.video_path, "Video path must be provided for encoding."
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device, args.output_path)
        torch.save(encoded_output, os.path.join(args.output_path, "encoded.pt"))
        decoded_output = decode_video(args.model_path, os.path.join(args.output_path, "encoded.pt"), dtype, device, args.output_path)
        save_video(decoded_output, args.output_path)

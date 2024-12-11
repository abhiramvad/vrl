import argparse
import torch
import imageio
import matplotlib.pyplot as plt
import numpy as np
from diffusers import AutoencoderKLCogVideoX
from torchvision import transforms
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import pandas as pd
import csv
from math import log10
import sys
import traceback
import gc

def compute_psnr(original_np, reconstructed_np):
    mse_val = mean_squared_error(original_np.flatten(), reconstructed_np.flatten())
    if mse_val == 0:
        return float('inf')
    pixel_max = 255.0
    psnr_val = 20 * log10(pixel_max / np.sqrt(mse_val))
    return psnr_val

def sample_videos(dataset_path, video_folder, num_videos_per_category=5, debug=False):
    dataset = pd.read_csv(dataset_path)

    # Filter dataset to include only videos present in the video_folder
    available_videos = set(os.listdir(video_folder))
    dataset = dataset[dataset["video"].isin(available_videos)]

    if debug:
        print(f"Filtered dataset to {len(dataset)} videos available in the video folder.")

    numeric_columns = [
        "aesthetic score",
        "motion score",
        "temporal consistency score",
        "fps",
        "seconds",
    ]

    for column in numeric_columns:
        dataset[f"{column}_bin"] = pd.cut(
            dataset[column], bins=3, labels=["low", "medium", "high"]
        )

    group_columns = ["camera motion"] + [f"{col}_bin" for col in numeric_columns]
    grouped = dataset.groupby(group_columns)

    # Sample videos
    sampled_videos = grouped.apply(
        lambda x: x.sample(min(len(x), num_videos_per_category), random_state=42)
    ).reset_index(drop=True)

    print("Sampled Videos Distribution:")
    print(sampled_videos[group_columns].value_counts())

    return sampled_videos

def visualize_and_save_tensor(tensor, output_dir, filename, title="Tensor Visualization"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tensor = tensor.to(dtype=torch.float32)

    if tensor.ndim == 4:
        tensor = tensor[0, :, :, :]
    if tensor.ndim == 3 and tensor.shape[0] > 3:
        tensor = tensor[0, :, :]
    tensor = tensor.cpu().numpy()

    tensor = (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor) + 1e-5)

    plt.imshow(tensor, cmap="viridis")
    plt.title(title)
    plt.axis("off")
    filepath = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

def encode_video(model, video_path, dtype, device):
    print("Encoding Video...")
    try:
        video_reader = imageio.get_reader(video_path, "ffmpeg")
        frames = [transforms.ToTensor()(frame) for frame in video_reader]
        video_reader.close()
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        raise

    frames_tensor = (
        torch.stack(frames)
        .to(device)
        .permute(1, 0, 2, 3)
        .unsqueeze(0)
        .to(dtype)
    )
    with torch.no_grad():
        encoded_result = model.encode(frames_tensor)
    encoded_frames = encoded_result[0].sample()
    print("Encoding frames completed.")
    print(f"Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
    print(f"Encoded frames shape: {encoded_frames.shape}, dtype: {encoded_frames.dtype}")

    # Note: original video is compressed, latent is raw tensor data
    original_size = os.path.getsize(video_path) / (1024 * 1024)
    latent_tensor = encoded_frames
    latent_size = (
        latent_tensor.numel()
        * torch.finfo(latent_tensor.dtype).bits
        / 8
        / (1024 * 1024)
    )

    print(f"Original (compressed) video size: {original_size:.4f} MB")
    print(f"Latent (raw tensor) size: {latent_size:.4f} MB (raw data, may be larger than compressed file)")
    return frames_tensor, encoded_frames, original_size, latent_size

def decode_video(model, encoded_frames, dtype, device):
    print("Decoding Video")
    with torch.no_grad():
        decoded_frames = model.decode(encoded_frames).sample
    print("Decoding completed.")
    return decoded_frames

def save_video(tensor, output_path):
    tensor = tensor.to(dtype=torch.float32)
    frames = tensor[0].permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)

    writer = imageio.get_writer(output_path, fps=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video saved at {output_path}")

def compute_aggregated_metrics(original_frames, decoded_frames, frame_sampling_interval=10):
    print("Computing aggregated metrics...")
    T = min(original_frames.shape[2], decoded_frames.shape[2])
    sampled_indices = range(0, T, frame_sampling_interval)
    orig_np = (original_frames.cpu().numpy() * 255).astype(np.uint8)
    dec_np = (decoded_frames.cpu().numpy() * 255).astype(np.uint8)

    sampled_orig = orig_np[0, :, list(sampled_indices), :, :]
    sampled_dec = dec_np[0, :, list(sampled_indices), :, :]

    sampled_orig_flat = sampled_orig.flatten()
    sampled_dec_flat = sampled_dec.flatten()

    mse = mean_squared_error(sampled_orig_flat, sampled_dec_flat)
    psnr_val = compute_psnr(sampled_orig, sampled_dec)

    ssim_values = []
    for idx in sampled_indices:
        o_frame = np.transpose(orig_np[0, :, idx], (1, 2, 0))
        d_frame = np.transpose(dec_np[0, :, idx], (1, 2, 0))
        try:
            ssim_score = ssim(o_frame, d_frame, multichannel=True, win_size=3)
        except ValueError:
            ssim_score = ssim(
                o_frame,
                d_frame,
                multichannel=True,
                win_size=min(o_frame.shape[0], o_frame.shape[1], 3),
            )
        ssim_values.append(ssim_score)

    avg_ssim = np.mean(ssim_values) if ssim_values else 0.0

    print(f"Computed metrics: MSE={mse}, SSIM={avg_ssim}, PSNR={psnr_val}")
    return mse, avg_ssim, psnr_val

def process_all_videos(
    model_path, sampled_videos_df, video_folder, output_folder, dtype, device, debug=False, frame_sampling_interval=10
):
    # Print how many videos and which videos are being considered
    print(f"Number of videos being considered: {len(sampled_videos_df)}")
    print("Videos under consideration:")
    for v in sampled_videos_df["video"].unique():
        print(v)
    sys.stdout.flush()

    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.enable_slicing()
    model.enable_tiling()

    os.makedirs(output_folder, exist_ok=True)
    plot_dir = os.path.join(output_folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    metrics_summary_path = os.path.join(output_folder, "metrics_summary.csv")
    csv_file_exists = os.path.isfile(metrics_summary_path)

    print(f"Metrics will be saved to: {os.path.abspath(metrics_summary_path)}")

    dataset_columns = list(sampled_videos_df.columns)
    custom_cols = ["mse", "ssim", "psnr", "original_size_mb", "latent_size_mb"]
    fieldnames = dataset_columns + custom_cols

    with open(metrics_summary_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not csv_file_exists:
            writer.writeheader()
            csvfile.flush()

        for idx, row in sampled_videos_df.iterrows():
            video_file = row["video"]
            video_path = os.path.join(video_folder, video_file)
            print(f"\n----------\nProcessing video ({idx+1}/{len(sampled_videos_df)}): {video_file}\n")
            sys.stdout.flush()

            original_frames = None
            encoded_frames = None
            decoded_frames = None

            try:
                original_frames, encoded_frames, original_size, latent_size = encode_video(
                    model, video_path, dtype, device
                )
                if debug:
                    visualize_and_save_tensor(
                        original_frames[0, :, 0],
                        plot_dir,
                        f"{video_file}_input",
                        "First Input Frame",
                    )
                    visualize_and_save_tensor(
                        encoded_frames[0, :, 0],
                        plot_dir,
                        f"{video_file}_latent",
                        "First Latent Frame",
                    )

                decoded_frames = decode_video(model, encoded_frames, dtype, device)
                if debug:
                    visualize_and_save_tensor(
                        decoded_frames[0, :, 0],
                        plot_dir,
                        f"{video_file}_decoded",
                        "First Decoded Frame",
                    )

                reconstructed_video_path = os.path.join(
                    output_folder, f"reconstructed_{video_file}"
                )
                save_video(decoded_frames, reconstructed_video_path)

                mse, avg_ssim, psnr_val = compute_aggregated_metrics(
                    original_frames, decoded_frames, frame_sampling_interval
                )

                video_metrics = row.to_dict()
                video_metrics["mse"] = mse
                video_metrics["ssim"] = avg_ssim
                video_metrics["psnr"] = psnr_val
                video_metrics["original_size_mb"] = original_size
                video_metrics["latent_size_mb"] = latent_size

                print(f"Aggregated Metrics for {video_file}: {video_metrics}")
                writer.writerow(video_metrics)
                csvfile.flush()

                print(f"Finished processing video: {video_file}")

            except MemoryError:
                print(f"MemoryError encountered while processing {video_file}. Attempting to free memory.")
                sys.stdout.flush()
                traceback.print_exc()
                break
            except Exception as e:
                print(f"Error processing video {video_file}: {e}")
                traceback.print_exc()
                continue
            finally:
                # Free memory
                del original_frames, encoded_frames, decoded_frames
                torch.cuda.empty_cache()
                gc.collect()

    print("Processing complete for all sampled videos.")
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos using CogVideoX with aggregated metrics and dataset info")
    parser.add_argument("--model_path", type=str, required=True, help="Path to CogVideoX model")
    parser.add_argument("--video_folder", type=str, required=True, help="Folder with videos")
    parser.add_argument("--output_folder", type=str, default=".", help="Output folder")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to video dataset CSV")
    parser.add_argument("--dtype", type=str, default="float16", help="Computation dtype ('float16' or 'bfloat16')")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device ('cuda' or 'cpu')")
    parser.add_argument("--num_videos_per_category", type=int, default=5, help="Number of videos per category")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--frame_sampling_interval", type=int, default=10, help="Sample every Nth frame for metrics")

    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    sampled_videos_df = sample_videos(
        args.dataset_path,
        args.video_folder,
        num_videos_per_category=args.num_videos_per_category,
        debug=args.debug,
    )

    process_all_videos(
        args.model_path,
        sampled_videos_df,
        args.video_folder,
        args.output_folder,
        dtype,
        device,
        debug=args.debug,
        frame_sampling_interval=args.frame_sampling_interval
    )

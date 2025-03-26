import os
import pickle
import re
import time
import warnings

import pandas as pd
import torch
import numpy as np
from PIL import Image
from calflops import calculate_flops
from cpuinfo import get_cpu_info
from torchvision import transforms
from medseg.util.path_builder import PathBuilder
from medseg.models.segmentors import (
    SegformerB3, InternImageUperNet_T, TransNeXtUperNet_Tiny,
    VWFormerMiTB3, VWFormerConvNextS, FCBFormer, HarDNetDFUS, SegNeXtL, FUSegNet, UNet, MISSFormer, HiFormerB
)

MODEL_MAPPING = {
    "TransNeXt": (TransNeXtUperNet_Tiny(in_size=512), "transnextupernet_tiny"),
    "InternImage": (InternImageUperNet_T(in_size=512), "internimageupernet_t"),
    "VWFormerMiTB3": (VWFormerMiTB3(in_size=512), "vwformermitb3"),
    "SegFormer": (SegformerB3(in_size=512), "segformerb3"),
    "VWFormerConvNeXtS": (VWFormerConvNextS(in_size=512), "vwformerconvnexts"),
    ###############
    "FCBFormer": (FCBFormer(in_size=512), "fcbformer"),
    "HarDNet-DFUS": (HarDNetDFUS(in_size=512), "hardnetdfus"),
    "SegNeXt": (SegNeXtL(in_size=512), "segnextl"),
    "FuSegNet": (FUSegNet(in_size=512), "fusegnet"),
    "UNet": (UNet(in_size=512, use_pretrained=True), "unet"),
    "MISSFormer": (MISSFormer(in_size=512, encoder_pretrained=True, operate_on_224=True), "missformer"),
    "HiFormer": (HiFormerB(in_size=512), "hiformerb"),

}


# This script benchmarks the inference time (avg, std) of different models on a set of images.
# In addition to inference time, it also calculates GMACs, Parameters, Throughput, and Peak GPU Memory Usage and
# saves the results to a CSV file as well as a LaTeX table.


def load_model(model_name, checkpoint_path, device):
    """Load a model from a checkpoint."""
    model = MODEL_MAPPING[model_name][0]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=512, antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")

    orig_width_px, orig_height_px = image.size

    if orig_height_px != orig_width_px:
        # Pad the image to make it square by adding zeros to the smaller side
        size = max(orig_width_px, orig_height_px)
        pad_w = (size - orig_width_px) // 2
        pad_h = (size - orig_height_px) // 2
        padding = (pad_w, pad_h, size - orig_width_px - pad_w, size - orig_height_px - pad_h)
        transform_pad = transforms.Pad(padding, fill=0)
        image = transform_pad(image)

    return transform(image).unsqueeze(0)  # Add batch dimension


def benchmark_model(model_name, model_ckpt_path, img_dir, num_images=10, warmup_iters=3, seed=42):
    """Benchmark inference time for a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpts = [os.path.join(model_ckpt_path, fold, "best_checkpoint.pt")
                   for fold in os.listdir(model_ckpt_path) if os.path.isdir(os.path.join(model_ckpt_path, fold))]

    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))])
    np.random.seed(seed)
    image_files = sorted(np.random.choice(image_files, num_images, replace=False))
    image_tensors = [preprocess_image(os.path.join(img_dir, img)).to(device) for img in image_files]

    fold_metrics_and_times = []
    for model_ckpt in model_ckpts:
        model = load_model(model_name, model_ckpt, device)

        # Compute FLOPs, MACs and Parameter Count
        img_shape = image_tensors[0].shape
        flops, macs, params = calculate_flops(model=model,
                                              input_shape=tuple(img_shape),
                                              output_precision=4,
                                              print_results=False,
                                              print_detailed=False)

        # Warm-up iterations
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = model(image_tensors[0])

        times = []
        # Benchmarking
        for img_tensor in image_tensors:
            torch.cuda.synchronize() if device.type == "cuda" else None
            # Quick and Dirty for Testing: Use random instead of images!
            # img_tensor = torch.randn(img_tensor.shape, device=device)
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(img_tensor)
            torch.cuda.synchronize() if device.type == "cuda" else None
            times.append(time.perf_counter() - start_time)

        total_time = sum(times) * 1000  # Convert to ms
        avg_inference_time = total_time / num_images

        # Compute Throughput (Images per Second)
        throughput = 1000 / avg_inference_time  # IPS (images per second)

        # Measure Peak GPU Memory Usage
        peak_memory = "N/A"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                _ = model(image_tensors[0])  # Run model once
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB

        metrics = {
            "MACs (G)": macs,  # flops.total() / 1e9,
            "Parameters (M)": params,  # / 1e6,
            "Inference Time (ms)": avg_inference_time,
            "Throughput (IPS)": throughput,
            "Peak GPU Memory (MB)": peak_memory
        }

        fold_metrics_and_times.append((metrics, times))

    times = [times for _, times in fold_metrics_and_times]
    times = np.array(times).flatten()  # .tolist()
    fold_metrics = [metrics for metrics, _ in fold_metrics_and_times]

    # Convert list of dicts to a dictionary of lists (ignoring "N/A" values)
    aggregated = {}
    for key in fold_metrics[0]:
        values = [fold[key] for fold in fold_metrics if fold[key] != "N/A"]
        aggregated[key] = np.array(values) if values else None  # None if all are "N/A"

    # Assert that MACs and Params are the same across all folds
    assert len(np.unique(aggregated['MACs (G)'])) == 1, "Not all elements are identical for 'MACs (G)'"
    assert len(np.unique(aggregated['Parameters (M)'])) == 1, "Not all elements are identical for 'Parameters (M)'"

    # Compute mean and std only for numeric values
    if round(np.mean(times) * 1000, 3) != np.mean(aggregated['Inference Time (ms)']).round(3):
        warnings.warn(f"The mean of all times should be equal to the mean of avg times but were "
                      f"{round(np.mean(times) * 1000, 3)} and {np.mean(aggregated['Inference Time (ms)']).round(3)}.")

    # Calculate mean and std of the inference times from raw times because
    # the std of the mean times is not the same as the std of the raw times
    avg_time = np.mean(times) * 1000  # Average inference time across num_images * num_folds images
    std_time = np.std(times) * 1000

    summary_metrics = {
        "MACs (G)": aggregated['MACs (G)'][0],
        "Parameters (M)": aggregated['Parameters (M)'][0],
        "Inference Time (ms)": f"{avg_time:.2f} ± {std_time:.2f}",
        "Throughput (IPS)": f"{aggregated['Throughput (IPS)'].mean():.2f}",
        # ± {aggregated['Throughput (IPS)'].std():.2f}",
        "Peak GPU Memory (MB)": f"{aggregated['Peak GPU Memory (MB)'].mean():.2f} ± {aggregated['Peak GPU Memory (MB)'].std():.2f}" if device.type == "cuda" else "N/A"
    }

    return summary_metrics, fold_metrics, times


if __name__ == "__main__":
    datasets = {
        "ukw": PathBuilder().root().add("data").add("datasets").add("ukw").add("images").build(),
        # "dfuc_22_test": PathBuilder().root().add("data").add("datasets").add("dfuc_22_test").build(),
        "cfu": PathBuilder().root().add("data").add("datasets").add("cfu").add("images").build(),
    }

    num_images = 300  # Number of images to test
    warmup_iters = 10  # Number of warm-up iterations
    seed = 42  # Random seed for image selection

    all_results = {}

    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        device_type = "cuda"
        gpu_info = torch.cuda.get_device_name(0)
        safe_gpu_name = re.sub(r'[^a-zA-Z0-9]', '_', gpu_info)  # Replace non-alphanumeric with "_"
        device_name = re.sub(r'_+', '_', safe_gpu_name).strip('_')
    else:
        device_type = "cpu"
        cpu_info = get_cpu_info()
        raw_cpu_name = cpu_info["brand_raw"]
        safe_cpu_name = re.sub(r'[^a-zA-Z0-9]', '_', raw_cpu_name)
        device_name = re.sub(r'_+', '_', safe_cpu_name)

    print(f"Running benchmarking on {num_images} images from each dataset on {device_name} ({device_type}).")

    for dataset, img_path in datasets.items():
        print(f"Running benchmarking on {dataset} dataset.")
        results = {}

        intermediate_filename = f"intermediate_benchmark_results_{dataset}_{device_type}_{device_name}.pkl"
        out_path = PathBuilder().root().out().add("benchmarking").add("extended_benchmarker").add("random").add(device_type) \
            .add(intermediate_filename).build()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        for model_name in MODEL_MAPPING.keys():
            model_ckpt_path = PathBuilder().root().out().add("k_fold_models").add(MODEL_MAPPING[model_name][1]).build()
            # benchmark_model returns a tuple of (summary_metrics, fold_metrics, times)
            print(f"Benchmarking model {model_name}")
            results[model_name] = benchmark_model(model_name, model_ckpt_path, img_path, num_images, warmup_iters, seed)

            # Save intermediate results to pickle
            with open(out_path, 'wb') as f:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

        # Save results to CSV
        csv_filename = f"benchmark_results_{dataset}_{device_type}_{device_name}.csv"
        out_path = PathBuilder().root().out().add("benchmarking").add("extended_benchmarker").add("random").add(device_type) \
            .add(csv_filename).build()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        model_summarys = {name: values[0] for name, values in results.items()}
        df = pd.DataFrame.from_dict(model_summarys, orient='index',
                                    columns=['MACs (G)', 'Parameters (M)', 'Inference Time (ms)',
                                             'Throughput (IPS)', 'Peak GPU Memory (MB)'])
        df.to_csv(out_path, index_label='Model')
        df.to_latex(out_path.replace(".csv", ".tex"))
        print(f"Benchmark results saved to {out_path}")

        all_results[dataset] = results

    # Compute mean across datasets, before save all results to pickle
    pickle_filename = f"all_benchmark_results_{device_type}_{device_name}.pkl"
    out_path = PathBuilder().root().out().add("benchmarking").add("extended_benchmarker").add("random").add(device_type)\
        .add(pickle_filename).build()
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)

    mean_results = {}
    for model_name in MODEL_MAPPING.keys():
        # Retrieve the summary metrics for each model across all datasets
        model_summary_metrics = [all_results[dataset][model_name][0] for dataset in datasets]
        model_avg_times = [float(dataset_id['Inference Time (ms)'].split(" ± ")[0]) for dataset_id in
                           model_summary_metrics]
        # Retrieve all raw times for each model across all datasets
        model_all_times = [all_results[dataset][model_name][2] for dataset in datasets]
        model_all_times = list(np.concatenate(model_all_times))
        model_all_times_in_ms = [time * 1000 for time in model_all_times]
        # Check that the mean of all times is equal to the mean of avg times
        if round(np.mean(model_all_times_in_ms).item(), 2) != np.mean(model_avg_times).round(2):
            warnings.warn(f"The mean of all times should be equal to the mean of avg times but were "
                          f"{round(np.mean(model_all_times_in_ms).item(), 2)} and {np.mean(model_avg_times).round(2)}.")

        # MACs and Parameters should be the same across all datasets
        assert len(np.unique([dataset_id['MACs (G)'] for dataset_id in model_summary_metrics])) == 1, \
            "Not all elements are identical for 'MACs (G)'"
        assert len(np.unique([dataset_id['Parameters (M)'] for dataset_id in model_summary_metrics])) == 1, \
            "Not all elements are identical for 'Parameters (M)'"

        # The IPS should be 1000 / avg_time
        avg_of_summary_ips = np.mean([float(metric['Throughput (IPS)']) for metric in model_summary_metrics]).item()
        if round(avg_of_summary_ips, 1) != (1000 / np.mean(model_all_times_in_ms)).round(1):
            warnings.warn(f"Throughput mismatch: {round(avg_of_summary_ips, 1)} vs "
                          f"{(1000 / np.mean(model_all_times_in_ms)).round(1)}.")

        # The mean of the peak GPU memory should be the mean of the peak GPU memory across all datasets
        # However the std for the peak GPU memory should be 0 for one dataset
        if device_type == "cpu":
            peak_gpu_memory_str = "N/A"
        else:
            model_peak_GPU_times = [float(dataset_id['Peak GPU Memory (MB)'].split(" ± ")[0])
                                    for dataset_id in model_summary_metrics]
            model_peak_GPU_times_std = [float(dataset_id['Peak GPU Memory (MB)'].split(" ± ")[1])
                                        for dataset_id in model_summary_metrics]

            if not all(v == 0 for v in model_peak_GPU_times_std):
                warnings.warn(f"Std of the peak GPU memory should be 0 for one dataset but were not.")

            peak_gpu_memory_str = f"{np.mean(model_peak_GPU_times):.2f} ± {np.std(model_peak_GPU_times):.2f}"

        mean_results[model_name] = (
            model_summary_metrics[0]['MACs (G)'],
            model_summary_metrics[0]['Parameters (M)'],
            f"{np.mean(model_all_times_in_ms):.2f} ± {np.std(model_all_times_in_ms):.2f}",
            f"{1000 / np.mean(model_all_times_in_ms):.2f}",
            peak_gpu_memory_str
        )

    # Save mean results to CSV
    mean_csv_filename = f"benchmark_results_across_all_datasets_{device_type}_{device_name}.csv"
    out_path = PathBuilder().root().out().add("benchmarking").add("extended_benchmarker").add("random").add(device_type)\
        .add(mean_csv_filename).build()
    df_mean = pd.DataFrame.from_dict(mean_results, orient='index',
                                     columns=['MACs (G)', 'Parameters (M)', 'Inference Time (ms)',
                                              'Throughput (IPS)', 'Peak GPU Memory (MB)'])

    # Do some formatting for the latex table
    df_mean['MACs (G)'] = df_mean['MACs (G)'].str.replace(" GMACs", "", regex=False).astype(float).round(2)
    df_mean['Parameters (M)'] = df_mean['Parameters (M)'].str.replace(" M", "", regex=False).astype(float).round(2)

    # Reorder columns
    df_mean = df_mean[['Parameters (M)', 'MACs (G)', 'Inference Time (ms)',
                       'Throughput (IPS)', 'Peak GPU Memory (MB)']]

    df_mean.to_csv(out_path, index_label='Model')
    df_mean.to_latex(out_path.replace(".csv", ".tex"))
    print(f"Mean benchmark results saved to {out_path}")

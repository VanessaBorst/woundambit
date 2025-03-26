import os
import json
import pandas as pd
import numpy as np

# Folder containing JSON files
DATA_FOLDER = "physician_data"
PREDICTIONS_FILE = "out/wound_analysis_results_largest.csv"


def load_json_files(data_folder):
    """Load all physician JSON files into a list of dictionaries."""
    data = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            with open(os.path.join(data_folder, filename), "r") as file:
                data.append(json.load(file))
    return data


def parse_wound_details_feedback(data):
    """Parse expert wound size annotations into a DataFrame and compute A_GT."""
    records = []
    for physician in data:
        name = physician["name"]
        for img_id, details in physician["image_feedback"].items():
            records.append({
                "physician": name,
                "image_id": img_id,
                # Convert cm to mm
                "height_expert": details["height"] * 10,
                "width_expert": details["width"] * 10
            })

    df = pd.DataFrame(records)

    # Ensure height is always the larger value
    df[['height_expert', 'width_expert']] = df.apply(
        lambda row: (max(row['height_expert'], row['width_expert']),
                     min(row['height_expert'], row['width_expert'])),
        axis=1,
        result_type='expand')

    # Create the size column in "H x W" format
    df["size"] = df["height_expert"].astype(int).astype(str) + " x " + df["width_expert"].astype(int).astype(str)

    # Pivot the table to get sizes per doctor
    df_pivot = df.pivot(index="image_id", columns="physician", values="size").reset_index()

    # Function to split "H x W" into separate height and width columns
    def _extract_dimensions(row):
        heights = []
        widths = []
        for expert in ["Astrid", "Caro", "Tassilo"]:
            h, w = map(int, row[expert].split(" x "))  # Convert to integers
            heights.append(h)
            widths.append(w)
        return pd.Series([heights, widths])

    # Function to compute inconsistency scores
    def compute_relative_deviation(values):
        """Compute relative deviation as (max-min)/median."""
        median = np.median(values)
        if median == 0:
            return float("inf")  # Avoid division by zero
        return (max(values) - min(values)) / median

    df_pivot[["heights", "widths"]] = df_pivot.apply(_extract_dimensions, axis=1)

    df_pivot["Rel_Dev_Height"] = df_pivot["heights"].apply(compute_relative_deviation)
    df_pivot["Rel_Dev_Width"] = df_pivot["widths"].apply(compute_relative_deviation)

    threshold_rel_dev = 0.5
    df_consistent_rel_dev = df_pivot[(df_pivot["Rel_Dev_Height"] <= threshold_rel_dev) &
                                     (df_pivot["Rel_Dev_Width"] <= threshold_rel_dev)]

    # Save to CSV
    df_pivot.to_csv("out/expert_size_estimates_per_image.csv", index=False)
    df_consistent_rel_dev.to_csv(
        f"out/consistent_expert_size_estimates_per_image_threshold_relative_dev_{str(threshold_rel_dev).replace('.', '_')}.csv")

    # Compute mean height, width & ground truth area (A_GT) per image
    mean_df = df.groupby("image_id").agg(
        height_GT=("height_expert", "mean"),
        width_GT=("width_expert", "mean"),
        # height_GT_variance=("height_expert", "var"),
        # width_GT_variance=("width_expert", "var"),
        height_GT_std=("height_expert", "std"),
        width_GT_std=("width_expert", "std")
    ).reset_index()
    mean_df["A_GT"] = mean_df["height_GT"] * mean_df["width_GT"]

    return mean_df


def parse_mask_feedback(data):
    """Parse mask feedback into a DataFrame."""
    records = []
    for physician in data:
        for img_id, models in physician["mask_feedback"].items():
            for model, feedback in models.items():
                records.append({
                    "image_id": img_id,
                    "model_name": model,
                    "judgment": feedback["judgment"]
                })
    return pd.DataFrame(records)


def parse_best_model_feedback(data):
    """Parse best model feedback into a DataFrame."""
    records = []
    for physician in data:
        for img_id, details in physician["best_model_feedback"].items():
            records.append({
                "image_id": img_id,
                "best_model": details["best_model"]
            })
    # Manually add best model selection of Caro (for the 2 images for which none of the six masks were considered good)
    records.append({"image_id": "Bild_09", "best_model": "InternImage"})
    records.append({"image_id": "Bild_17", "best_model": "VWFormerConvNeXtS"})
    return pd.DataFrame(records)


def calculate_metrics(wound_size_df, mask_feedback_df, best_model_df, predictions_df):
    """Compute CMA, ECR, MAE, MAPE, MPA, and MEA."""

    # Clinical Mask Approval (CMA)
    good_masks_per_model = mask_feedback_df[mask_feedback_df["judgment"] == "Good"].groupby("model_name").size()
    total_masks_per_model = mask_feedback_df.groupby("model_name").size()
    cma_df = (good_masks_per_model / total_masks_per_model).reset_index()
    cma_df.columns = ["model_name", "CMA"]
    cma_df["CMA"] *= 100  # Convert to percentage

    # Images with at least one bad mask
    bad_masks = mask_feedback_df[mask_feedback_df["judgment"] == "Bad"].sort_values("image_id")
    # Difficult images: images with at least one bad mask
    difficult_images = bad_masks["image_id"].unique()
    images_without_bad_masks = set(mask_feedback_df["image_id"].unique()) - set(difficult_images)

    # Expert Choice Rate (ECR)
    number_selections_per_model = best_model_df.groupby("best_model").size()
    ecr_df = (number_selections_per_model / len(best_model_df)).reset_index()
    ecr_df.columns = ["model_name", "ECR"]
    ecr_df["ECR"] *= 100  # Convert to percentage

    # For better interpretation per image:
    # best_per_image_df = best_model_df.groupby("image_id")["best_model"].apply(lambda x: list(set(x))).reset_index()

    # Merge predictions with expert annotations
    # Merge on Image ID to align ground truth areas (A_GT) with predictions
    merged_df = predictions_df.merge(wound_size_df[['image_id', 'A_GT', 'width_GT', 'height_GT']], on='image_id')
    merged_df['diff_width'] = merged_df['width_mm'] - merged_df['width_GT']
    merged_df['diff_height'] = merged_df['height_mm'] - merged_df['height_GT']

    # Define functions for metrics
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100  # Convert to percentage

    def compute_metrics(group):
        """Compute size retrieval metrics for each model."""
        return pd.Series({
            "MAE_Height": mae(group["height_GT"], group["height_mm"]),
            "MAE_Width": mae(group["width_GT"], group["width_mm"]),
            "MAPE_Height": mape(group["height_GT"], group["height_mm"]),
            "MAPE_Width": mape(group["width_GT"], group["width_mm"]),
            "Mean_Predicted_Height": group["height_mm"].mean(),
            "Mean_Predicted_Width": group["width_mm"].mean(),
            "SD_Predicted_Height": group["height_mm"].std(),
            "SD_Predicted_Width": group["width_mm"].std(),
            "Mean_Expert_Height": group["height_GT"].mean(),  # Should be the same across models
            "Mean_Expert_Width": group["width_GT"].mean(),  # Should be the same across models
            "SD_Expert_Height": group["height_GT"].std(),  # Should be the same across models
            "SD_Expert_Width": group["width_GT"].std()  # Should be the same across models
        })

    # Apply the function to compute metrics for each model
    # Relative Deviation <= 0.5
    image_ids_to_keep = pd.read_csv("out/consistent_expert_size_estimates_per_image_threshold_relative_dev_0_5.csv")["image_id"].tolist()

    filtered_df = merged_df[merged_df['image_id'].isin(image_ids_to_keep)]
    results_df = filtered_df.groupby("model_name").apply(compute_metrics)

    ####### Prepare Latex Tables for Copying ###############

    custom_order = ['TransNeXt', 'InternImage', 'VWFormerMiTB3', 'SegFormer', 'VWFormerConvNeXtS',  'Ensemble']
    results_df = results_df.reindex(custom_order)

    results_df["MAE_Height"] = results_df["MAE_Height"].round(1)
    results_df["MAE_Width"] = results_df["MAE_Width"].round(1)
    results_df["MAPE_Height"] = results_df["MAPE_Height"].round(1)
    results_df["MAPE_Width"] = results_df["MAPE_Width"].round(1)

    # Format all numerical columns to strings with one decimal place
    results_df = results_df.applymap(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)

    # Combine Mean and SD into a single column
    results_df["Predicted Height"] = results_df["Mean_Predicted_Height"].round(1).astype(str) + \
                                     " ± " + results_df["SD_Predicted_Height"].round(1).astype(str)
    results_df["Predicted Width"] = results_df["Mean_Predicted_Width"].round(1).astype(str) + \
                                    " ± " + results_df["SD_Predicted_Width"].round(1).astype(str)

    # Keep only relevant columns and rename for LaTeX output
    df_final = results_df[["Predicted Height", "MAE_Height", "MAPE_Height",
                           "Predicted Width", "MAE_Width", "MAPE_Width"]]
    df_final.index.name = "Model"  # Set index name for table header

    # Export to LaTeX
    latex_output = df_final.to_latex(column_format="lcccccc", escape=False)

    # Save to file
    with open(f"out/size_retrieval_results.tex", "w") as f:
        f.write(latex_output)

    print(f"Results saved to out/size_retrieval_results.tex")

    ####### End Latex ###############

    print("Processing finished")


# Load and parse data
os.makedirs("out/", exist_ok=True)
data = load_json_files(DATA_FOLDER)
wound_size_df = parse_wound_details_feedback(data)
mask_feedback_df = parse_mask_feedback(data)
best_model_df = parse_best_model_feedback(data)

# Load model predictions from CSV
predictions_df = pd.read_csv(PREDICTIONS_FILE)

# Ensure height is always the larger value
predictions_df[['height_mm', 'width_mm']] = predictions_df.apply(
    lambda row: (max(row['height_mm'], row['width_mm']),
                 min(row['height_mm'], row['width_mm'])),
    axis=1,
    result_type='expand')

# # Do some summarizing for the paper
# predictions_df["Area_in_cm2"] = predictions_df["area_mm2"] / 100
#
# # Format all numerical columns to strings without decimal place except the area
# predictions_df.loc[:, predictions_df.columns != 'Area_in_cm2'] = \
#     predictions_df.loc[:, predictions_df.columns != 'Area_in_cm2'].applymap(lambda x: f"{round(x):.0f}"
#     if isinstance(x, (int, float)) else x)
#
# # Apply rounding with one decimal place to the area
# predictions_df['Area_in_cm2'] = predictions_df['Area_in_cm2'].apply(lambda x: f"{round(x, 1):.1f}")
#
# # Combine Mean and SD into a single column
# predictions_df["Size"] = predictions_df["height_mm"].astype(str) + " x " \
#                          + predictions_df["width_mm"].astype(str)
#
# # Keep only relevant columns and rename for LaTeX output
# final_df = predictions_df[["image_id", "model_name", "Size", "Area_in_cm2"]]
# final_df.index.name = "ID"  # Set index name for table header
#
# final_df["model_name"] = pd.Categorical(final_df["model_name"],
#                                         ["VWFormerConvNeXtS", "InternImage", "VWFormerMiTB3",
#                                          "SegFormer", "TransNeXt", "Ensemble"])
# final_df.sort_values(by=['image_id', 'model_name'], inplace=True)
#
# # Export to LaTeX
# latex_output = final_df.to_latex(column_format="lcc", escape=False)
#
# # Save to file
# with open(f"out/predicted_sizes_and_area_for_paper.tex", "w") as f:
#     f.write(latex_output)

# Compute metrics
calculate_metrics(wound_size_df, mask_feedback_df, best_model_df, predictions_df)

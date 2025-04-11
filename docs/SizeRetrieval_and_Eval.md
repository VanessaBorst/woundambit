# Size Retrieval Tool

## Size Retrieval

This tool analyzes pairs of 1) images containing our reference object with ArUco markers and 2) wound masks to accurately determine wound dimensions. It uses ArUco markers for calibration, computing a pixel-to-millimeter ratio to ensure precise measurements.

### Key Idea

The tool identifies the longest and shortest diagonals of a wound mask contour using a sweeping perpendicular line method. 
It also supports multi-wound mode, either measuring the largest wound (default) or summing dimensions across multiple wounds.

### Usage

1. Ensure the images are named starting with "Image" (e.g., "Image\_01.jpg") and masks end with "_mask.png" 
   (e.g., "Image\_ID\_modelX\_mask.png"). 
   We include the image-mask pairs used for our publication as an example for the naming scheme.
2. Place your images and masks in the "image\_mask\_data" directory.
3. Adapt the default parameters in the `size_retrieval.py` file (if desired):
   - `DATA_DIR`: Directory containing images and masks (images as jpg, masks as png)
   - `MULTI_WOUND_MDOE`: "largest" or "sum"
   - `OUTPUT_CSV`: Path to the output CSV file for results
   - `OUTPUT_DIR`: Directory to store annotated images
4. Run the script (from within the `src/main/medseg/tools/deployment` directory or 
   adapt the paths in the script to match your working directory, if necessary):

```bash
python size_retrieval.py 
```

### Output
The above command processes all image-mask pairs in `image_mask_data`.
It saves the results to a CSV `wound_analysis_results_<mode>.csv` with the following structure:
`image_id,model_name,area_mm2,width_mm,height_mm`, which contains the predicted wound area in mm², width in mm, and
height in mm for each image-model pair.
Moreover, annotated images that illustrate the size retrieval are saved in `OUTPUT_DIR`.

## Size Retrieval Evaluation

The evaluation of the size retrieval is performed using the `manual_assessment_eval.py` script.
It can only be run after the size retrieval tool has been executed and requires the manual expert annotations that
we created for the publication and provide in the `physician_data` folder. 
Before metrics calculation, the script filters the data based on the relative deviation between the expert annotations.

### Usage
To reproduce our reported results, it can be run with the following command:

```bash
python manual_assessment_eval.py
```

### Output
The above command produces three output files that are saved into the `out` folder:
1. `size_retrieval_results.tex`: 
   A LaTeX table containing the results of the size retrieval evaluation, including the Clinical Mask Approval (CMA) , Expert Choice Rate (ECR), mean absolute error(MAE), and mean absolute percentage error (MAPE) for each model, alongside the mean predicted height and width (MPH, MPW).
2. `expert_size_estimates_per_image.csv`:
   A CSV file containing the expert size estimates for all images.
3. `consistent_expert_size_estimates_per_image_threshold_relative_dev_0_5.csv`:
   A CSV file containing the expert size estimates for each image, filtered by a relative deviation threshold of 0.5.



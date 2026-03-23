# Project Title: TESSERA Framework Customisation

**By Gijs van den Dool — [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/gvdool/) | [GitHub](https://github.com/GvdDool)**

Satellite imagery has enabled global mapping since the launch of Landsat 1 in 1972. But assigning each pixel just one label is no longer sufficient for many applications. Foundation models like TESSERA work differently: instead of fixed classes, they create detailed 128-dimensional embeddings that capture the full spectral and temporal signature of each place.

Foundation models can learn from many types of data, including Landsat time series spanning several decades. Modern systems like TESSERA use both Sentinel-1 SAR and Sentinel-2 optical images. Combining these sources provides better spatial detail, more frequent updates, and cloud-free images, all of which help improve embedding quality in tough conditions.

Turning these rich, multi-source datasets into consistent, high-quality embeddings across large areas enables many applications, such as land-cover analysis, change detection, and crop monitoring. These tasks need little fine-tuning and do not require retraining the main model.

The main challenge is making this work at scale.
**GitHub Repository for this work:** Colab Notebooks and ReadMe

-----

## Customising TESSERA's framework

TESSERA is an open-source foundation model developed at the University of Cambridge ([github.com/ucam-eo/tessera](https://github.com/ucam-eo/tessera)). The repository is well-documented and structured: the codebase is organised, preprocessing scripts are parameterised, and the inference pipeline is logically separated from data preparation.

The main challenge is not the code, but the computing resources. Without a dedicated high-performance computing (HPC) setup, running the full pipeline at once can quickly exhaust available memory and storage when working with years of satellite data. The Cambridge team is building global-scale embeddings as a 'global carpet' (Model-as-Data), but the huge amount of data, measured in petabytes, means rolling this out worldwide takes a long time.

To address this and mitigate the waiting time, I structured the pipeline as a set of discrete, self-contained functions, each invoked within a double loop over areas and years. This approach transforms a potentially fragile monolithic script into a controllable, resumable process: if a single year or tile fails, only that specific combination needs to be rerun.

To manage this process, I implemented a Master-Worker architecture within Google Colab. The 'Master' notebook handles the setup and processing schedule, while the 'Worker' is a collection of modular functions. Executing both within a shared kernel enables seamless state management and allows scaling analysis without the overhead of frequent file I/O.

-----

## The Pipeline: Fifteen Steps

The pipeline divides into two phases: seven configuration steps that run once per session and eight processing steps that execute within a double loop for every area-year combination.

### Phase 1: Configuration (Steps 1 to 7)

Sets up a reproducible environment by connecting to storage, getting the needed code, installing dependencies, and standardising paths and permissions. The model is also prepared for fast inference by moving checkpoints locally and applying targeted patches. These steps ensure that all subsequent processing runs reliably within the Colab runtime.

  * **Step 1: Preparing the Google Colab environment.** This step connects the Colab runtime to persistent storage, where the repository, checkpoints, and final outputs are stored. Google Colab Pro with an L4 GPU and high-RAM runtime is used to align with the requirements specified by the TESSERA team (the Dask data manipulation tasks (and workers) need the additional compute [Disclaimer: I didn't test the methods on the Google Free Tier.]
  * **Step 2: Clone the TESSERA repository.** The specified branch is pulled from GitHub into the Drive-mounted directory, making preprocessing and inference scripts available to the pipeline. If the repository is already present, the active branch is verified, and the clone step is skipped.
  * **Step 3: Install dependencies.** All required Python packages, including rasterio, xarray, stackstac, pyproj, pystac-client, planetary-computer, and others, are installed into the Colab environment. This step is separate from the import step to ensure packages are available before any module-level imports are attempted.
  * **Step 4: Configure project paths.** All working directories, including data, temporary files, checkpoints, and output, are resolved into a single configuration object that is passed through the pipeline. Local scratch space (/content/) is used for intermediate input and output, while only final outputs are written back to Google Drive. Having extra personal storage is a main reason for choosing Colab and Google Drive.
  * **Step 5: Localise the model checkpoint.** The latest TESSERA model weights (about 7.5 GB) are saved on a personal Google Drive. The chosen checkpoint is then copied to the local runtime. This avoids file path issues, ensures the pipeline uses the correct model, and speeds up inference loading.
  * **Step 6: Update preprocessing scripts.** Two specific fixes were made to the TESSERA code. The first change is to update the inference script to use the correct Python path for Colab. The second adds cloud class 10 (thin cirrus) to the Sentinel-2 invalid-pixel mask. Thin cirrus clouds (SCL class 10) are partly see-through but are often masked in optical processing because they can affect surface reflectance and related measurements. In places like West Africa, where clouds are common, removing these pixels improves the quality of composites and analysis, but it also means fewer scenes are usable when cloud cover is below 20%.
  * **Step 7: Set file permissions.** The Rust-compiled stacking binaries and shell scripts are marked as executable. This step is required for every new Colab session because the Drive-mounted filesystem does not preserve Unix permissions across sessions.

### Phase 2: Main Processing Loop (Steps 8 to 15)

Runs for each area and year, combining optical and radar data to make georeferenced embeddings for analysis. The steps include defining a region of interest, downloading Sentinel-1 and Sentinel-2 images, stacking and patching the data, running the model, and exporting the results. Storage is managed efficiently in local scratch space. This approach keeps processing consistent and repeatable across locations and times. All functions match the scripts in the GitHub repository.

  * **Step 8: Generate Region of Interest (ROI).** The central coordinates and tile size for each area are converted into a georeferenced single-band GeoTIFF using the correct UTM projection. This ROI sets the boundaries for all later spatial steps and the final output. To keep memory use reasonable and processing stable, a 5×5 km area (500×500 cells, like the TESSERA chip) is used by default. This usually takes about 8 minutes in the chosen Colab environment.
  * **Step 9: Download Sentinel-2.** The Microsoft Planetary Computer STAC catalogue is searched for Sentinel-2 (S2) images with low cloud cover for the ROI and year. The images are saved directly to the local scratch space. By default, only scenes with up to 20% cloud cover are included, which helps avoid very cloudy images in overcast areas. While TESSERA can handle some clouds, using mostly clear images helps each patch produce high-quality embeddings, especially in cloudy regions.
  * **Step 10: Download Sentinel-1.** The same catalogue is used to get SAR backscatter data from the Microsoft Planetary Computer Sentinel-1-rtc collection. Sentinel-1 (S1) images are not affected by clouds, so they are a useful addition to optical images in tough weather. The SAR images from MPC are already terrain-corrected, so topographic effects are handled during preprocessing. Both ascending and descending orbits are requested, but only ascending images are available for this area. This may slightly affect data quality due to viewing angles and terrain, but about 30 images per year still provide a strong basis for analysis.
  * **Step 11: Stack.** The TESSERA Rust-compiled stackers for S1 and S2 are executed in parallel, consolidating the downloaded GeoTIFFs into numpy arrays organised by band, date, and orbit state. The resulting stacked arrays are stored entirely in local scratch space.
  * **Step 12: Retile.** The stacked arrays are divided into 40×40 pixel patches for inference. A 5 km tile at 10 m resolution yields 169 patches. The retiler also generates a small ROI mask for each patch, which is subsequently used to address edge cases at tile boundaries.
  * **Step 13: Inference.** The TESSERA model is applied to all patches using the localised checkpoint. This step is computationally intensive and requires a GPU runtime (L4 or A100). Processing 169 patches typically takes 6 to 10 minutes. Each patch produces a 40×40×128 embedding array.
  * **Step 14: Stitch and export.** The 169 patch embeddings are combined into one 500×500×128 spatial array, georeferenced to the original ROI. This array is saved as a GeoTIFF in the area's output folder on Drive. The 128 MB file works with standard GIS tools and can be used by any downstream classifier.
  * **Step 15: Cleanup.** All temporary files, such as downloads, stacks, patches, and representations, are deleted from local scratch space to free up storage before starting the next area and year. The final outputs on Drive are not affected.

-----

## Results: Six Years of Embeddings Over a Single Area

To validate the pipeline, I generated embeddings for a consistent 5×5 km ROI across six consecutive years (2020–2025). Each year produces a 500×500×128 GeoTIFF — where every pixel encodes the full spectral-temporal signature of that location across a year of Sentinel-1 and Sentinel-2 acquisitions.

To visualise this high-dimensional latent space, Principal Component Analysis (PCA) was fitted once across the combined six-year dataset and used to reduce each year to three primary components, mapped to RGB channels. By fitting the PCA globally, we ensure that colours are comparable across the entire time series: a shift in hue reflects a genuine change in the landscape rather than a statistical artefact.

The resulting animation reveals an immediate and interpretable structure:

  * **Stable features:** The river running through the western edge appears as a consistent deep blue across all six years—a stable spectral-temporal signature that the model encodes reliably without any supervision.
  * **Dynamic shifts:** The urban area to the east shifts in colour noticeably over time, reflecting (real) changes in the built environment and surrounding land use.
  * **Fine texture:** The agricultural areas show differences between crop types, fallow periods, and agroforestry patterns in impressive detail.

No classifier or labels were used. The patterns in these images are entirely derived from the learned embeddings, providing a sensitive, unsupervised way to track changes in the landscape.

PCA was fitted once across all six years to ensure a consistent colour reference frame. No classifier or labels were applied.

**Note:** The Colab Notebook to run the tests is stored in the GitHub repository: `V4_TESSERA_Master.ipynb`, which links to `V4_TESSERA_Worker.ipynb` (both files are required to reproduce the work).

-----

## Refinement

Scaling is still a challenge, but this modular setup makes it easier to develop broad, systematic views and improves our understanding of year-over-year changes.

The table below illustrates this directly. For this test area, deliberately chosen in a region with persistent cloud cover, only 2023 has enough cloud-free Sentinel-2 scenes to meet the minimum threshold for meaningful embeddings. In most years, clear acquisitions cluster in the dry season, meaning that year-over-year comparisons carry an inherent seasonal bias that cannot be resolved without denser temporal coverage.

**Table 1. Cloud-filtered Sentinel-2 acquisition dates for the Soubre test tile (max. 20% cloud cover, 2020–2025)**

The threshold for cloud presence in a scene is set to 20%. For traditional machine learning tasks, this is already considered high, but the TESSERA documentation suggests accepting up to 90–100% cloud cover, relying on pixel-level SCL masking to filter out individual cloudy pixels within each scene. In principle, relaxing the threshold would increase temporal coverage and reduce seasonal sampling bias.

However, in West Africa, cloud cover during the rainy season is not only dense but also persistent, accompanied by shading, haze, and atmospheric scattering that degrade surface reflectance even in nominally cloud-free pixels. In this context, lowering the threshold may increase scene count without meaningfully improving embedding quality, so perhaps a conservative 20% filter could be the safer default.

Using a strict tile-based filter has a clear effect, as shown in Figure 1. Not all years are equally represented, so there are big differences between years. Only 2023 had enough cloud-free images (about 10% of all passes) to make strong embeddings. In other years, the few images from the dry season create a built-in seasonal bias.

**Table 2. Minimum cloud cover threshold required to reach sufficient scene count per year (bold = first threshold meeting ≥7 scenes), Soubre test tile 2020–2025.**

To understand whether relaxing the cloud threshold could improve temporal coverage without degrading embedding quality, a systematic experiment was conducted across 16 thresholds from 15% to 90%. This is the scene-level filter; it reflects cloud cover across the full 110×110 km Sentinel-2 tile, not just the 5×5 km ROI. The processor applies a secondary ROI-level filter, rejecting chips where more than 5% of pixels are flagged as invalid by the Scene Classification Layer (SCL). For scenes with low cloud cover, all chips pass, but as scene-level cloud tolerance increases, more chips are rejected at the internal SCL stage. For 2023, at 90% tolerance, 53 scenes were prepared from 73 available observations, with 11 chips rejected by the internal check.

**Figure 2. Sentinel-2 scene selection frequency by date, Soubre test tile 2023, across 16 cloud cover thresholds (15%–90%).** Bar height indicates how many thresholds are included in each date. Blue = successfully downloaded; red = rejected by internal SCL check.

-----

## Cloud Threshold Experiment

Full embeddings were generated at each of the 16 thresholds using an incremental per-date download approach. In the Colab environment, Dask task scheduling is unstable under high memory pressure, leading to premature task terminations that cascade across subsequent dates. To mitigate this, each date was processed in an isolated subprocess with a fresh Dask cluster, avoiding the cascade failures that occurred when processing full-year ranges with high cloud tolerances.

The resulting embeddings were compared using three complementary analyses.

1.  **Consecutive threshold delta curve.** Mean absolute difference and cosine similarity were calculated between each pair of consecutive thresholds: t15 to t20, t20 to t25, and so on (t05, t10 are excluded because they have the same scene selection as t15, and t95 and t100 are removed as the internal check would reject most of the chips with this much cloud cover). The delta curve shows a large jump from t15 to t20 (cosine similarity increases from 0.942 to 0.984), then remains flat from t25 onward at around 0.989. The Mean Absolute Difference follows a similar pattern but is mirrored, dropping from 2.3167 (t15) to 0.9949 (t20), with the absolute minimum at t025 when going to t030 (0.6651). Adding more scenes this year does not improve results; the key point for this ROI is at t20 (with 7 scenes).

**Figure 3. Embedding change between consecutive cloud cover thresholds, Soubre test tile 2023.** Top: mean absolute difference. Bottom: mean cosine similarity. The elbow at t15 to t20 is the only significant transition.

2.  **Pairwise cosine similarity matrix.** Comparing all 16 threshold pairs shows three clear embedding groups: t20 to t40 (dry season only), t45 to t65 (dry season plus shoulder season), and t70 to t90 (full year, including rainy season). Inside each group, embeddings are very similar (\>0.982). Between-group similarity drops to 0.960-0.975, which is meaningful but not large. t15 stands apart from all other thresholds because it has only 4 scenes, which also explains the large differences in animation across years, with few images (2021, 2024, 2025). Even though these years were not tested, they are likely to show the same level of isolation as the 2023-t15 selection.

**Figure 4. Pairwise cosine similarity matrix across 16 cloud cover thresholds.** Three distinct embedding clusters are visible, separated at approximately t45 and t70.

3.  **PCA variance explained per threshold.** PCA was fitted once across all 16 threshold embeddings combined, then applied to each threshold individually to measure variance explained by the first three principal components. Variance jumps from 11.5% at t15 to approximately 15% at t20, then flattens almost completely through t90 (16.6%). Only 1.6 percentage points of additional variance are gained across 14 threshold steps beyond t20.

**Figure 5. PCA variance explained per threshold, Soubre test tile 2023.** Variance flattens at t20, confirming that no significant structural improvement occurs beyond that point.

**Note:** The Colab Notebook to run the tests is stored in the GitHub repository: `LinkedInFigures.ipynb`

-----

## Operational Recommendation

The three analyses converge on the same conclusion: for this ROI, 7 scenes at 20% cloud cover capture approximately 90% of the embedding structure achievable with 42 scenes at 90% cloud cover. However, as the acquisition table shows, most years do not reach 7 scenes at 20%. The following reproducible workflow was developed to determine the operational threshold for any ROI:

1.  **Find the best year** — the year reaching sufficient scene count at the strictest threshold. For this ROI: 2023, reaching 7 scenes at 20%.
2.  **Run the cloud threshold experiment on the best year** — generate embeddings across all thresholds, compute the delta curve and PCA variance, and identify the elbow point. For this ROI: elbow at t20, 7 scenes.
3.  **Determine the ROI-specific minimum viable scene count** — the scene count at the elbow point. For this ROI: 7 scenes.
4.  **Apply to all other years** — find the minimum threshold needed per year to reach the minimum scene count. For this ROI, all years reach 7+ scenes at 30% cloud cover.
5.  **Select the operational threshold** — the lowest threshold at which all years meet the minimum, with comfortable headroom. For this ROI: 35%, giving ≥10 scenes per year across all six years.

At 35%, processing time drops significantly compared to running at 90%, about 10 scenes per year instead of 42, while still remaining within the stable t20 to t40 embedding group.

-----

## Visual Confirmation

In 2024, only 3 scenes are found in the ROI at 20% cloud cover, which is insufficient to capture the landscape's optical components, with sensor artefacts dominating the embedding. This is not an isolated case; 2025 at 20% yields only 2 scenes, and the degradation is visible across the entire top row of Figure 6: as scene count drops from 7 (2023) to 3 (2024) to 2 (2025), spatial structure progressively breaks down within the same shared colour space.

The bottom row tells a different story: at 35% cloud cover, 2024 gains 8 scenes and 2025 gains 9 scenes, all three years show consistent spatial structure and comparable colour signatures, with field boundaries, road networks, and vegetation patterns clearly resolved. The choice of threshold does not just affect the scene count; it determines whether the embedding is structurally comparable across years at all.

**Figure 6. TESSERA embeddings for 2023, 2024, and 2025 at t020 (top row) and t035 (bottom row)**, projected into a shared PCA colour space fitted across all six images. Within t020, spatial structure degrades as scene count drops from 7 (2023) to 3 (2024) to 2 (2025). Within t035, all three years show consistent structure and comparable colour signatures despite different scene counts. Same area, same model, same colour space.

-----

## Outlook

My primary objective is to use these notebooks to build a regional embedding archive spanning various areas and years. While the official global TESSERA outputs are being produced and validated by the Cambridge team, this 'on-demand' approach allows projects to train downstream Earth observation models early — without the need for high-performance computing (HPC) or waiting for global rollouts.

The architecture is designed for easy expansion:

  * **Scalability:** Adding a new study area is as simple as changing a single coordinate setting.
  * **Flexibility:** Every step—from cloud-cover thresholds to SAR orbit selection—can be adjusted independently.
  * **Throughput:** The double-loop structure supports parallel execution across multiple Colab sessions for users processing entire countries.

By structuring the pipeline this way, the workflow transitions from experimental scripts to a production-ready data-processing system.

The notebooks are available at [github.com/GvdDool](https://github.com/GvdDool). I welcome feedback and contributions, and I am excited to see how others use these 128-dimensional signatures for their own environmental projects\!

\#EarthObservation \#TESSERA \#OpenScience \#RemoteSensing \#MachineLearning \#Sustainability

-----

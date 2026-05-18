# GeoAlign: Online Calibration via Latent Directional Consistency

This repository contains the official implementation of the ICML 2026 paper **"GeoAlign: Geometric Rollout Curation for Robust LLM Reinforcement Learning"**.

## 1. Environment Setup

Our codebase is built upon Trinity-RFT. Please follow the steps below to set up the environment.

### 1.1. Basic Dependencies

Navigate to the Trinity-RFT directory and install the package in editable mode along with the required dependencies:

```bash
cd Trinity-RFT
pip install -e ".[dev]"
pip install flash-attn==2.8.1
```

### 1.2. vLLM Modification

**GeoAlign** requires access to the **last-layer hidden state** of the generated sequences to compute directional consistency. The standard `vllm` library does not expose these hidden states during generation. To enable this efficient "one-pass" extraction, we have modified the `vllm` library.

1. **Version Requirement:** We use `vllm==0.9.1`.
2. **Apply Changes:** We provide a diff file containing the necessary modifications to the vLLM source code.
   * **Diff Location:** `geoalign/hidden_vllm_0.9.1/changes.diff`
   * Run the following commands to apply the patch:

```bash
# Activate your environment
conda activate <your_trinity_env>

# Apply the patch
VLLM_PATH=$(python -c "import importlib.util; print(importlib.util.find_spec('vllm').submodule_search_locations[0])")
patch -p1 -d $VLLM_PATH < geoalign/hidden_vllm_0.9.1/changes.diff
```

3. **Verification:** After applying the changes, run the provided test script to ensure hidden states are being returned correctly:

```bash
python geoalign/hidden_vllm_0.9.1/test_hidden_vllm.py
```

---

## 2. Usage

GeoAlign is implemented as an **Experience Buffer operator** within the Trinity-RFT framework. The core logic is located at:
`Trinity-RFT/trinity/buffer/operators/filters/outlier_reward_filter.py`.

### 2.1. Architecture: Decoupled Projector Training

The Trinity-RFT framework locks GPU resources during the main RL training loop. To allow the GeoAlign projector (a lightweight MLP) to train effectively without resource conflicts or blocking the main process, we decouple the projector training using a **Watcher Process**.

* **Main Process (Trinity):** Generates rollouts, saves preference data to a specified path, and signals a request for projector training.
* **Watcher Process:** Runs in the background. It monitors for training signals, loads the data, trains the projector on the GPU, saves the updated model, and signals completion back to Trinity.

### 2.2. Running the Code

**Step 0: Prepare the Reward Model API (HH-RLHF only)**

This step is only required for reinforcement learning on the HH-RLHF dataset. Since reward computation requires calling a large external reward model (ArmoRM-Llama3-8B-v0.1 in our experimental setup), we first deploy it as an API service so that Trinity-RFT can query it during training. A deployment script is provided:

```bash
cd Trinity-RFT/geoalign/RM_API_construct
CUDA_VISIBLE_DEVICES=7 uvicorn rm_api_ArmoRM:app --host 0.0.0.0 --port 6007 --workers 1
```

To verify the deployment, run:

```bash
python geoalign/RM_API_construct/request_try.py
```

The API interface used during RL training is `class APIRewardFn` in `Trinity-RFT/trinity/common/rewards/reward_fn.py`. Please ensure that `self.API_URL = 'http://127.0.0.1:6007/score'` points to a reachable endpoint before launching training.

**Step 1: Start the Projector Watcher**

Before launching the main training loop, you must start the watcher process. Ensure the `inter_data_path` variable in the script points to your desired directory for intermediate model checkpoints (e.g., `./preference_classifier_ckpt`).

```bash
cd ./Trinity-RFT
python geoalign/train_watcher.py
```

> **Note:** Keep this process running in the background or in a separate terminal window throughout the training.

**Step 2: Launch the RL Training**

Once the watcher is active, start the Ray cluster and the Trinity training job.

```bash
ray start --head

# Example 1: For Mathematical Reasoning (DAPO dataset)
trinity run --config yamls/dapo.yaml

# Example 2: For HH-RLHF
trinity run --config yamls/hh_rlhf.yaml
```

---

## 3. Configuration Guidelines

You will need to adjust the YAML configuration files (e.g., `yamls/dapo.yaml`) to match your local paths and specific model settings. The relevant parameters for GeoAlign are located under `data_processor -> experience_pipeline -> operators`.

### Parameter Explanation

Below is an example configuration for the `outlier_reward_filter` operator:

```yaml
data_processor:
  experience_pipeline:
    operators:
      - name: "outlier_reward_filter"
        args:
          # Dimension of the policy's last hidden layer.
          # This value corresponds to "hidden_size" in the model's config.json.
          # Set to 2048 for Qwen3-1.7B; set to 2560 for Qwen3-4B.
          input_hidden_state_dim: 2048

          # Dimension of the projected latent space (d').
          output_hidden_dim: 512

          rollout_num: ${algorithm.repeat_times}

          # (Kappa) Upper bound fraction for potential outlier candidates.
          outlier_rank_ratio: 0.2

          # (Alpha) KDE sensitivity for detecting the anomaly score boundary.
          peak_pdf_alpha: 0.05

          reward_reshape_type: max_random  # Options: max_random / remove / replace_with_group_mean

          # Path to the training script used by the watcher process.
          subprocess_dir: Trinity-RFT/trinity/buffer/operators/filters/outlier_reward_filter_train_subprocess.py

          # Directory for intermediate checkpoints.
          # MUST match the 'inter_data_path' defined in 'train_watcher.py'.
          classifier_model_save_path: ./preference_classifier_ckpt
```
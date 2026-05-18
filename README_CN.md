# GeoAlign: Geometric Rollout Curation for Robust LLM Reinforcement Learning

本仓库包含 ICML 2026 论文 **《GeoAlign: Geometric Rollout Curation for Robust LLM Reinforcement Learning》** 的官方实现。

## 1. 环境配置

本代码基于 Trinity-RFT 框架构建，请按照以下步骤配置环境。

### 1.1. 基础依赖

进入 Trinity-RFT 目录，以可编辑模式安装包及所需依赖：

```bash
cd Trinity-RFT
pip install -e ".[dev]"
pip install flash-attn==2.8.1
```

### 1.2. vLLM 修改

**GeoAlign** 需要在生成过程中获取模型最后一层的隐状态，用于计算方向一致性。标准 `vllm` 库不暴露这些隐状态，为此我们对其进行了修改，以实现高效的"单次前向传播"提取。

1. **版本要求：** 使用 `vllm==0.9.1`。
2. **应用修改：** 我们提供了一个 diff 文件，包含对 vLLM 源码的全部必要修改。
   * **文件位置：** `geoalign/hidden_vllm_0.9.1/changes.diff`
   * 运行以下命令应用补丁：

```bash
# 激活你的环境
conda activate <your_trinity_env>

# 应用补丁
VLLM_PATH=$(python -c "import importlib.util; print(importlib.util.find_spec('vllm').submodule_search_locations[0])")
patch -p1 -d $VLLM_PATH < geoalign/hidden_vllm_0.9.1/changes.diff
```

3. **验证：** 应用修改后，运行测试脚本确认隐状态能够被正确返回：

```bash
python geoalign/hidden_vllm_0.9.1/test_hidden_vllm.py
```

---

## 2. 使用方法

GeoAlign 以 **Experience Buffer 算子** 的形式集成在 Trinity-RFT 框架中，核心逻辑位于：
`Trinity-RFT/trinity/buffer/operators/filters/outlier_reward_filter.py`

### 2.1. 架构：解耦的投影器训练

Trinity-RFT 框架在主 RL 训练循环期间会独占 GPU 资源。为了让 GeoAlign 的投影器（一个轻量级 MLP）能够有效训练而不产生资源冲突或阻塞主进程，我们通过 **监听进程（Watcher Process）** 将投影器训练与主训练流程解耦。

* **主进程（Trinity）：** 生成 rollout，将偏好数据保存到指定路径，并发送投影器训练请求信号。
* **监听进程：** 在后台运行，监听训练信号，加载数据，在 GPU 上训练投影器，保存更新后的模型，并向主进程发送完成信号。

### 2.2. 运行步骤

**第 0 步：部署奖励模型 API（仅 HH-RLHF 场景需要）**

此步骤仅在 HH-RLHF 数据集上进行强化学习时需要。由于该场景的奖励计算需要调用大型外部奖励模型（我们的实验中使用 ArmoRM-Llama3-8B-v0.1），需要先将其部署为 API 服务，以便 Trinity-RFT 在训练过程中调用。我们提供了一份部署脚本：

```bash
cd Trinity-RFT/geoalign/RM_API_construct
CUDA_VISIBLE_DEVICES=7 uvicorn rm_api_ArmoRM:app --host 0.0.0.0 --port 6007 --workers 1
```

部署完成后，可以运行以下脚本验证是否成功：

```bash
python geoalign/RM_API_construct/request_try.py
```

RL 训练中调用奖励模型的接口为 `Trinity-RFT/trinity/common/rewards/reward_fn.py` 中的 `class APIRewardFn`，请在启动训练前确认其中的 `self.API_URL = 'http://127.0.0.1:6007/score'` 指向可正常访问的端点。

**第 1 步：启动监听进程**

在启动主训练循环之前，必须先启动监听进程。请确认脚本中的 `inter_data_path` 变量指向你希望存放中间模型检查点的目录（例如 `./preference_classifier_ckpt`）。

```bash
cd ./Trinity-RFT
python geoalign/train_watcher.py
```

> **注意：** 请在整个训练过程中保持该进程在后台或独立终端窗口中持续运行。

**第 2 步：启动 RL 训练**

监听进程启动后，启动 Ray 集群并运行 Trinity 训练任务。

```bash
ray start --head

# 示例 1：数学推理（DAPO 数据集）
trinity run --config yamls/dapo.yaml

# 示例 2：HH-RLHF
trinity run --config yamls/hh_rlhf.yaml
```

---

## 3. 配置说明

你需要根据本地路径和模型设置调整 YAML 配置文件（如 `yamls/dapo.yaml`）。GeoAlign 相关参数位于 `data_processor -> experience_pipeline -> operators` 下。

### 参数说明

以下是 `outlier_reward_filter` 算子的配置示例：

```yaml
data_processor:
  experience_pipeline:
    operators:
      - name: "outlier_reward_filter"
        args:
          # 策略模型最后一层隐状态的维度。
          # 该值对应模型 config.json 中的 "hidden_size"。
          # Qwen3-1.7B 设为 2048；Qwen3-4B 设为 2560。
          input_hidden_state_dim: 2048

          # 投影后隐空间的维度（d'）。
          output_hidden_dim: 512

          rollout_num: ${algorithm.repeat_times}

          # (Kappa) 异常候选样本的比例上限。
          outlier_rank_ratio: 0.2

          # (Alpha) KDE 异常分数边界检测的敏感度。
          peak_pdf_alpha: 0.05

          reward_reshape_type: max_random  # 可选：max_random / remove / replace_with_group_mean

          # 监听进程使用的训练脚本路径。
          subprocess_dir: Trinity-RFT/trinity/buffer/operators/filters/outlier_reward_filter_train_subprocess.py

          # 中间检查点保存目录。
          # 必须与 train_watcher.py 中的 inter_data_path 保持一致。
          classifier_model_save_path: ./preference_classifier_ckpt
```
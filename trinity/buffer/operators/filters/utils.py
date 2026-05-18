
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, normalize

import time
import os

def wait_for_file(path: str, timeout: int = 60, interval: int = 5):
    """
    Block until the file appears, raise an exception on timeout.
    """
    start_time = time.time()

    while True:
        if os.path.exists(path):
            return

        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Timed out waiting for file ({timeout}s): {path}"
            )

        time.sleep(interval)
        
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import skew
    

def detect_anomalies_kde(eid_scores_dict, outlier_rank_ratio=0.1, peak_pdf_alpha = 0.05):
    """
    Dynamically detect anomalous samples using KDE (Kernel Density Estimation).
    :param eid_scores_dict: dictionary of {id: score}
    :param outlier_rank_ratio: upper bound ratio to prevent excessive filtering (safety cap)
    """
    print(f'peak_pdf_alpha = {peak_pdf_alpha}')
    items = list(eid_scores_dict.items())
    scores = np.array([x[1] for x in items])
    n_samples = len(scores)

    if n_samples < 100:  # 样本太少时 KDE 不可靠，退回到保守逻辑
        return [],0
    # skewscore = skew(scores)
    # if skewscore < 1.2: 
    #     print(1)
    #     # return [], 0

    # 1. 拟合 KDE
    kde = gaussian_kde(scores, bw_method='scott')
    
    # 2. 在分数范围内创建网格进行评估
    x_grid = np.linspace(min(scores), max(scores), 200)
    pdf = kde(x_grid)

    # 3. 寻找动态阈值：寻找 PDF 下降到一定程度的“尾部起点”
    # 策略：找到 PDF 曲线下降最剧烈后的平缓点（或者简单取峰值的 10% 处作为长尾边界）
    peak_pdf = np.max(pdf)
    # 找到所有 PDF 小于峰值 10% 的点中，位于右侧（高分侧）的部分
    tail_indices = np.where((pdf < peak_pdf_alpha * peak_pdf) & (x_grid > x_grid[np.argmax(pdf)]))[0]
    
    if len(tail_indices) > 0:
        kde_threshold = x_grid[tail_indices[0]]
    else:
        kde_threshold = np.mean(scores) + 2 * np.std(scores) # 兜底 fallback

    # 4. 初步筛选
    anomalous_candidates = [
        (eid, score) for eid, score in items if score >= kde_threshold
    ]
    
    anomalous_candidates.sort(key=lambda x: x[1], reverse=True)
    max_allowed_num = int(n_samples * outlier_rank_ratio)
    
    final_anomalies = anomalous_candidates[:max_allowed_num]
    print(f'len(kde) = {len(anomalous_candidates)}')
    print(f'len(final) = {max_allowed_num}')
    
    return [x[0] for x in final_anomalies],len(anomalous_candidates)


class PreferenceClassifier(nn.Module):
    """
    Preference classifier network.
    Learns to identify the direction of h_diff vectors.
    
    Architecture: Linear -> ReLU -> Linear (output logits)
    """
    def __init__(self, input_dim: int, hidden_layer_size: int = None):
        super().__init__()
        
        self.hidden_layer_size = input_dim // 2 if hidden_layer_size is None else hidden_layer_size
            
        print(f"Input({input_dim}) -> Hidden({self.hidden_layer_size}) -> Output(1)")


        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, self.hidden_layer_size),
            nn.ReLU(),
        )
        
        self.classifier_head = nn.Linear(self.hidden_layer_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier_head(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class PreferencePairDataset(Dataset):
    """
    Dataset for loading preference data with arbitrary numbers of responses.
    Each sample returns a tuple of (h_better, h_worse, weight).
    weight = score difference (always positive)
    """
    def __init__(self, raw_data: List[Dict[str, Any]]):
        self.pairs = []  

        for item in raw_data:
            responses = item.get('responses', [])
            pref_matrix = np.array(item.get('preference_matrix_rm', []))

            if len(responses) == 0 or pref_matrix.size == 0:
                continue

            n = len(responses)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pref_matrix[i, j]

                    if diff == 0:
                        continue

                    if diff > 0:  
                        h_better = responses[i]["hidden_state"]
                        h_worse = responses[j]["hidden_state"]
                    else:          
                        h_better = responses[j]["hidden_state"]
                        h_worse = responses[i]["hidden_state"]

                    weight = abs(diff)  
                    self.pairs.append((h_better, h_worse, weight))

        if len(self.pairs) == 0:
            raise ValueError("No valid preference pairs generated. Please check the input data.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        h_better, h_worse, weight = self.pairs[idx]

        if isinstance(h_better, np.ndarray):
            h_better = torch.from_numpy(h_better).float()
        elif isinstance(h_better, torch.Tensor):
            h_better = h_better.detach().clone().float()

        if isinstance(h_worse, np.ndarray):
            h_worse = torch.from_numpy(h_worse).float()
        elif isinstance(h_worse, torch.Tensor):
            h_worse = h_worse.detach().clone().float()

        if isinstance(weight, torch.Tensor):
            weight = weight.detach().clone().float()
        else:
            weight = torch.tensor(weight, dtype=torch.float32)

        return h_better.squeeze(), h_worse.squeeze(), weight

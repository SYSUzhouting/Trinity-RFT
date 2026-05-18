import numpy as np
from tqdm import tqdm
import pickle
import torch
import numpy as np
import uuid
from trinity.buffer.operators import EXPERIENCE_OPERATORS, ExperienceOperator
from trinity.common.experience import Experience, group_by

import sqlite3, os, re, pickle,sys, time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, normalize

from .utils import PreferenceClassifier, PreferencePairDataset, wait_for_file, detect_anomalies_kde
from torch import Tensor
import subprocess
import copy

@EXPERIENCE_OPERATORS.register_module("outlier_reward_filter")
class OutlierRewardFilter(ExperienceOperator):

    def __init__(self, 
                 input_hidden_state_dim: int = 2048,
                 output_hidden_dim: int = 512,
                 outlier_rank_ratio: float = 0.2,
                 peak_pdf_alpha: float = 0.05,
                 rollout_num: int = 8,
                 reward_reshape_type: str = 'remove',
                 subprocess_dir: str = 'Trinity-RFT/trinity/buffer/operators/filters/outlier_reward_filter_train_subprocess.py',
                 classifier_update_lr: float = 1e-4,
                 classifier_train_min_epochs: int = 50, 
                 classifier_train_max_epochs: int = 100,
                 classifier_train_early_stop_acc: float = 0.85,
                 classifier_model_save_path: str = './tpt'
                 ):
        
        self.input_hidden_state_dim = input_hidden_state_dim
        self.output_hidden_dim = output_hidden_dim
        self.rollout_num = rollout_num
        self.reward_reshape_type = reward_reshape_type

        self.subprocess_dir = subprocess_dir
        self.classifier_update_lr = classifier_update_lr
        self.classifier_train_min_epochs = classifier_train_min_epochs 
        self.classifier_train_max_epochs = classifier_train_max_epochs
        self.classifier_train_early_stop_acc = classifier_train_early_stop_acc
        self.classifier_model_save_path = classifier_model_save_path

        self.outlier_rank_ratio = outlier_rank_ratio
        self.peak_pdf_alpha = peak_pdf_alpha

        self.preference_classifier = PreferenceClassifier(input_dim=self.input_hidden_state_dim, hidden_layer_size=self.output_hidden_dim)
        
        self.data4training = pd.DataFrame(columns=["prompt", "response", "reward", "hidden_state"])
        self.all_experience_data_num = 0
        self.experience_round = 0

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        self.experience_round = self.experience_round + 1

        records = []
        for exp in exps:
            records.append({
                "prompt": getattr(exp, "prompt_text", None),
                "response": getattr(exp, "response_text", None),
                "reward": getattr(exp, "reward", None),
                "hidden_state": self.load_hidden(exp),
            })

        new_df = pd.DataFrame(records)
        self.data4training = pd.concat([self.data4training, new_df], ignore_index=True)
        self.all_experience_data_num = self.all_experience_data_num + len(exps)
        

        if len(exps) == 0:

            if len(exps) > 0:
                records = []

            for exp in exps:
                exp.reward_hidden_state = []
            filtered_exps = exps
            metrics = {"filtered_count": len(exps) - len(filtered_exps)}

        else: 
            raw_data = self.load_data(self.data4training)  # list

            self.preference_classifier = self.train_binary_classifier_for_hidden_space(
                subprocess_dir=self.subprocess_dir,
                raw_data=raw_data, 
                round_id=self.experience_round,
                model=self.preference_classifier,
                min_epochs=self.classifier_train_min_epochs, 
                max_epochs=self.classifier_train_max_epochs,
                early_stop_acc=self.classifier_train_early_stop_acc,
                batch_size=1024,
                learning_rate = self.classifier_update_lr,
                test_split = 0.1,
                model_save_path=self.classifier_model_save_path
            )


            self.data4training = pd.DataFrame(columns=["prompt", "response", "reward", "hidden_state"])
        
            updated_hidden_grouped_exps = self.exps2groups(exps)

            v_proto = self.compute_prototypes_binary(updated_hidden_grouped_exps)

            all_similarity_matrices, all_outlier_masks = self.detect_outliers_fast(updated_hidden_grouped_exps, v_proto)
            
            if self.reward_reshape_type == 'remove':
                extracted_eids = self.extract_anomaly_eids(all_similarity_matrices, updated_hidden_grouped_exps, replace_with_group_mean='remove')
                
                for exp in exps:
                    exp.reward_hidden_state = []
                filtered_exps = [exp for exp in exps if exp.eid not in extracted_eids]
                metrics = {"filtered_count": len(exps) - len(filtered_exps)}

            elif self.reward_reshape_type == 'replace_with_group_mean':
                extracted_eids_dict_list = self.extract_anomaly_eids(all_similarity_matrices, updated_hidden_grouped_exps, replace_with_group_mean='replace_with_group_mean')
                
                for exp in exps:
                    exp.reward_hidden_state = []
                eid_to_new_reward = {str(item['eid']): item['new_reward'] for item in extracted_eids_dict_list}
                
                filtered_exps = []
                replaced_count = 0
                for exp in exps:
                    if str(exp.eid) in eid_to_new_reward:
                        exp.reward = eid_to_new_reward[str(exp.eid)]
                        replaced_count += 1
                    filtered_exps.append(exp)
                metrics = {"replaced_count": replaced_count}

            elif self.reward_reshape_type == 'max_random':
                extracted_eids_dict_list = self.extract_anomaly_eids(
                    all_similarity_matrices, 
                    updated_hidden_grouped_exps, 
                    replace_with_group_mean='max_random'
                )

                
                eid_to_exp_obj = {str(exp.eid): exp for exp in exps}

                anomaly_to_target_id = {str(item['eid']): str(item['to_target_eid']) for item in extracted_eids_dict_list}

                filtered_exps = []
                replaced_count = 0

                for exp in exps:
                    eid_str = str(exp.eid)
                    if eid_str in anomaly_to_target_id:
                        target_eid_str = anomaly_to_target_id[eid_str]
                        target_exp_obj = eid_to_exp_obj[target_eid_str]
                        
                        filtered_exps.append(copy.deepcopy(target_exp_obj))
                        replaced_count += 1
                    else:
                        filtered_exps.append(exp)

                metrics = {"replaced_count": replaced_count}

        return filtered_exps, metrics
    
    def load_hidden(self, exp:Experience) -> Tensor:
        return getattr(exp, "policy_last_tokens_hidden", None)[-1]

    def exps2groups(self, exps: List[Experience]) -> List:
        records = []
        for exp in exps:
            records.append({
                "prompt": getattr(exp, "prompt_text", None),
                "response": getattr(exp, "response_text", None),
                "reward": getattr(exp, "reward", None),
                "hidden_state": self.load_hidden(exp),
                "eid": getattr(exp, "eid", None),
            })
        new_df = pd.DataFrame(records)

        groups_data_with_eid = self.load_data(new_df, use_eid = True)

        return groups_data_with_eid

    def load_data(self, raw_df: pd.DataFrame, use_eid: bool = False) -> List:
        raw_data = []
        rollout_num = self.rollout_num


        raw_df['group_id'] = (raw_df['prompt'] != raw_df['prompt'].shift()).cumsum()
        group_sizes = raw_df.groupby('group_id').size()
        valid_groups = group_sizes[group_sizes == rollout_num].index
        df_full = raw_df[raw_df['group_id'].isin(valid_groups)].copy()
        groups = df_full.groupby('prompt')


        for prompt_value, group_df in tqdm(groups, total=len(groups), desc="Processing groups"):

            all_items = []

            for _, row in group_df.iterrows():
                response = row['response']
                reward = row['reward']
                hidden_state = row['hidden_state']

                try:
                    conv = self.parse_qwen_chat_template(prompt_value, response)
                    input_history = conv[:-1]  # list
                    response_text = conv[-1]['content']  # str
                except:
                    input_history = ''
                    response_text = response

                if use_eid:
                    all_items.append({
                        "response": response_text,
                        "reward": reward,
                        "hidden_state": hidden_state,
                        "eid": row["eid"]
                    })
                else:
                    all_items.append({
                        "response": response_text,
                        "reward": reward,
                        "hidden_state": hidden_state,
                    })

            rewards = np.array([r["reward"] for r in all_items])
            # Constructs an NxN matrix where matrix[i][j] = reward[i] - reward[j].
            # If matrix[i][j] > 0, response[i] is preferred over response[j].
            preference_matrix = rewards[:, None] - rewards[None, :]  

            data_item = {
                "prompt": prompt_value,
                "input_history": input_history,
                "responses": all_items,
                "preference_matrix_rm": preference_matrix, 
            }

            raw_data.append(data_item)

        # time.sleep(3)

        return raw_data

    def parse_qwen_chat_template(self, text: str, response: str) -> List[dict]:
        text = text.strip()

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        pattern = re.compile(r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>", re.DOTALL)
        matches = pattern.findall(text)

        messages = []
        for role, content in matches:
            messages.append({
                "role": role.strip(),
                "content": content.strip()
            })
            
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        messages.append({"role": "assistant", "content": response})
        
        return messages

    def train_binary_classifier_for_hidden_space(self,
                                                subprocess_dir: str,
                                                raw_data: List, 
                                                round_id: int,
                                                model: PreferenceClassifier,
                                                min_epochs: int = 10, 
                                                max_epochs: int = 100,
                                                early_stop_acc: float = 0.85,
                                                batch_size: int = 512, 
                                                learning_rate: float = 1e-4,
                                                test_split: float = 0.1,
                                                model_save_path: str = "") -> Tuple[PreferenceClassifier,float]:
        
        os.makedirs(model_save_path, exist_ok=True)
        SIGNAL_PATH = os.path.join(model_save_path, "start_hidden_classifier_train.signal")
        
        uid = str(uuid.uuid4())[:8]
        
        session_dir = os.path.join(model_save_path, f"exp_{uid}_round_{round_id}")
        os.makedirs(session_dir, exist_ok=True)

        data_path = os.path.join(session_dir, "raw_data.pkl")
        model_path = os.path.join(session_dir, "hidden_space_classifier.pth")
        done_path = os.path.join(session_dir, "hidden_space_classifier.done")
        SH_PATH = os.path.join(session_dir, "run_train.sh")


        print(f"[{uid} | Round {round_id}] Saving data to: {session_dir}")
        with open(data_path, "wb") as f:
            pickle.dump(raw_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        sh_content = f"""#!/usr/bin/env bash
        python \\
        {subprocess_dir} \\
        --model_save_path "{session_dir}" \\
        --output_hidden_dim {str(self.output_hidden_dim)} \\
        --min_epochs {str(min_epochs)} \\
        --max_epochs {str(max_epochs)} \\
        --early_stop_acc {str(early_stop_acc)} \\
        --batch_size {str(batch_size)} \\
        --learning_rate {str(learning_rate)} \\
        --test_split {str(test_split)}

        """

        with open(SH_PATH, "w") as f:
            f.write(sh_content)

        os.chmod(SH_PATH, 0o755)

        print(f"[main] generate sh: {SH_PATH}")

        # 发信号（只写 sh 路径）
        with open(SIGNAL_PATH, "w") as f:
            f.write(SH_PATH)

        print("[main] Send Training signal.")

        wait_for_file(done_path, timeout=60, interval=5)

        print("[main] Model file detected, starting to load...")
        
        state_dict = torch.load(
            model_path,
            map_location="cpu"
        )

        model.load_state_dict(state_dict)
        model.eval()

        print(f"Successfully loaded model weights. Latent space dimension: {model.hidden_layer_size}")

        return model

    def compute_prototypes_binary(self, data_list) -> np.ndarray:
        vpair_list = []

        for item in data_list:
            responses = item.get("responses", [])
            if len(responses) < 2:
                continue

            hidden_list = []
            reward_list = []
            for r in responses:
                h = r.get("hidden_state")
                if h is None:
                    continue
                if isinstance(h, np.ndarray):
                    h_np = h.astype(np.float32)
                elif hasattr(h, "detach"):
                    h_np = h.detach().to(torch.float32).cpu().numpy()
                else:
                    continue
                hidden_list.append(h_np)
                reward_list.append(float(r.get("reward", 0.0)))

            n = len(hidden_list)
            if n < 2:
                continue

            for i in range(n):
                for j in range(n):
                    if reward_list[i] > reward_list[j]:
                        diff = hidden_list[i] - hidden_list[j]
                        vpair_list.append(diff)

        if len(vpair_list) == 0:
            return None

        vpair_array_ = np.stack(vpair_list)

        vpair_array_ = torch.from_numpy(vpair_array_)
        
        with torch.no_grad():
            vpair_array = self.preference_classifier.extract_features(vpair_array_)

        vpair_array = vpair_array.numpy()  # (N, d)

        norms = np.linalg.norm(vpair_array, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)

        vpair_array = vpair_array / norms      # (N, d)
        vprototype_center = np.mean(vpair_array, axis=0)  # (d,)

        print(f"✅ Total valid preference pairs loaded: {len(vpair_list)}")
        print(f"vprototype_center shape: {vprototype_center.shape}")

        return vprototype_center

    def detect_outliers_fast(self, data, v_proto, save_path=None, metric="cosine", tau=0.3, use_weight=False, return_type="matrix") -> Tuple[List[np.ndarray],List[np.ndarray]]:
        
        print(f"\n{'='*60}")
        print(f"Starting outlier detection for {len(data)} items...")

        v_proto = np.array(v_proto, dtype=np.float32)
        v_proto_norm = v_proto / (np.linalg.norm(v_proto) + 1e-8)

        all_similarity_matrices = []
        all_outlier_masks = []

        for item_idx, item in enumerate(data):
            responses = item.get("responses", [])
            n = len(responses)
            if n < 2:
                all_similarity_matrices.append(None)
                all_outlier_masks.append(None)
                continue

            hidden_list, reward_list = [], []
            for r in responses:
                h = r.get("hidden_state")
                if h is None:
                    continue
                if isinstance(h, np.ndarray):
                    h_np = h.astype(np.float32)
                elif hasattr(h, "detach"):
                    h_np = h.detach().to(torch.float32).cpu().numpy()
                else:
                    continue
                hidden_list.append(h_np)
                reward_list.append(float(r.get("reward", 0.0)))

            n = len(hidden_list)
            if n < 2:
                all_similarity_matrices.append(None)
                all_outlier_masks.append(None)
                continue

            hidden_arr = np.stack(hidden_list, axis=0)  # (n, d)
            diffs_ = hidden_arr[:, None, :] - hidden_arr[None, :, :]  # (n, n, d)
            
            diffs_ = torch.from_numpy(diffs_)
            with torch.no_grad():
                diffs = self.preference_classifier.extract_features(diffs_)
            diffs = diffs.numpy()

            rewards = np.array(reward_list, dtype=np.float32)
            reward_diff = rewards[:, None] - rewards[None, :]  # (n, n)

            mask_pos = reward_diff > 0

            diffs_norm = diffs / (np.linalg.norm(diffs, axis=2, keepdims=True) + 1e-8)
            cos_sim = np.dot(diffs_norm, v_proto_norm)  # (n, n)

            if use_weight:
                weight = np.maximum(reward_diff, 0)
                weighted_cos_sim = cos_sim * weight
            else:
                weighted_cos_sim = cos_sim

            outlier_mask = np.zeros_like(cos_sim, dtype=bool)
            outlier_mask[mask_pos] = cos_sim[mask_pos] < tau
            
            all_similarity_matrices.append(weighted_cos_sim)
            all_outlier_masks.append(outlier_mask)

            if (item_idx + 1) % 500 == 0:
                print(f"Processed {item_idx + 1} / {len(data)} samples...")

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump({
                    "similarity_matrices": all_similarity_matrices,
                    "outlier_masks": all_outlier_masks
                }, f)
            print(f"Saving result to: {save_path}")

        print(f"Outlier detection completed. Total samples processed: {len(data)}")
        return all_similarity_matrices, all_outlier_masks

    def extract_anomaly_eids(self,
            all_similarity_matrices: List[np.ndarray],
            updated_hidden_grouped_exps: List[Dict[str, Any]],
            replace_with_group_mean: str = 'remove'
        ) -> List:
            eid_accumulated_scores = {}
            eid_counts = {}
            uid_to_obj_mapping = {}
            
            for sim_matrix, item in zip(all_similarity_matrices, updated_hidden_grouped_exps):
                if sim_matrix is None:
                    continue
                
                responses = item.get("responses", [])
                rewards = np.array([float(r.get("reward", 0.0)) for r in responses])
                
                mask_pos = (rewards[:, None] - rewards[None, :]) > 0

                # mask_pos
                rows, cols = np.where(mask_pos)
                
                for r, c in zip(rows, cols):
                    penalty = 1.0 - sim_matrix[r, c]
                    
                    for idx in [r, c]:
                        eid_obj = responses[idx]["eid"]
                        u_key = eid_obj.uid 
                        
                        eid_accumulated_scores[u_key] = eid_accumulated_scores.get(u_key, 0.0) + penalty
                        
                        eid_counts[u_key] = eid_counts.get(u_key, 0) + 1

                        if u_key not in uid_to_obj_mapping:
                            uid_to_obj_mapping[u_key] = eid_obj

            if not eid_accumulated_scores:
                return []

            eid_avg_scores_dict = {
                u_key: eid_accumulated_scores[u_key] for u_key in eid_accumulated_scores
            }

            sorted_items = sorted(eid_avg_scores_dict.items(), key=lambda x: x[1], reverse=True)
            scores_array = np.array([x[1] for x in sorted_items])
            

            anomaly_uids, kde_num = detect_anomalies_kde(dict(sorted_items), self.outlier_rank_ratio, peak_pdf_alpha = self.peak_pdf_alpha)

            anomaly_uids_set = set(anomaly_uids) 

            if replace_with_group_mean == 'remove':
                return [uid_to_obj_mapping[uid] for uid in anomaly_uids]

            elif replace_with_group_mean == 'replace_with_group_mean':
                results = []
                for item in updated_hidden_grouped_exps:
                    group_resps = item.get("responses", [])
                    
                    safe_rewards = [
                        float(r['reward']) for r in group_resps 
                        if r['eid'].uid not in anomaly_uids_set
                    ]
                    mean_val = np.mean(safe_rewards) if safe_rewards else 0.0
                    
                    for r in group_resps:
                        if r['eid'].uid in anomaly_uids_set:
                            results.append({
                                'eid': r['eid'], 
                                'new_reward': mean_val
                            })
                return results

            elif replace_with_group_mean == 'max_random':
                results = []
                for item in updated_hidden_grouped_exps:
                    group_resps = item.get("responses", [])
                    
                    candidates = [
                        r for r in group_resps 
                        if r['eid'].uid not in anomaly_uids_set
                    ]
                    
                    if not candidates:
                        continue 
                        
                    candidate_scores = torch.tensor([float(r['reward']) for r in candidates])
                    max_score = torch.max(candidate_scores)
                    
                    weights = torch.where(candidate_scores == max_score, 0.5, 0.1)
                    
                    for r in group_resps:
                        if r['eid'].uid in anomaly_uids_set:
                            sampled_idx = torch.multinomial(weights, num_samples=1).item()
                            target_resp = candidates[sampled_idx]
                            
                            results.append({
                                'eid': r['eid'], 
                                'to_target_eid': target_resp['eid'] 
                            })

                return results
                
            return [uid_to_obj_mapping[uid] for uid in anomaly_uids]
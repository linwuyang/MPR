import hashlib
import random
import torch
from collections import deque

class GlobalReplay:
    def __init__(self, max_size=3000):
        self.max_size = max_size  # 最大行人数
        self.buffer = []          # 存储(obs, pred)元组
        self.current_peds = 0
        self.seen_hashes = set()
        self.hash_list = []       # 哈希列表
        self.seq_start_ends = []  # 新增：存储每个序列的[start, end]信息

    def _get_sequence_hash(self, obs_segment, pred_segment):
        """生成轨迹片段的唯一哈希（支持 CUDA 张量）"""
        obs_np = obs_segment.cpu().detach().numpy()
        pred_np = pred_segment.cpu().detach().numpy()
        combined = obs_np.tobytes() + pred_np.tobytes()
        return hashlib.md5(combined).hexdigest()

    def add(self, batch_data, y_hat_rel):
        """逐个添加序列，空间不足时随机淘汰旧数据"""
        _, _, obs_traj_rel, pred_traj_gt_rel, _, _, batch_seq_start_end = batch_data
        
        pred_logits = y_hat_rel
        
        for (start, end) in batch_seq_start_end:
            obs_segment = obs_traj_rel[:, start:end, :]
            pred_segment = pred_traj_gt_rel[:, start:end, :]
            logits_segment = pred_logits[:, start:end, :]
            num_peds = obs_segment.shape[1]
        
            # 生成哈希并检查重复
            seq_hash = self._get_sequence_hash(obs_segment, pred_segment)
            if seq_hash in self.seen_hashes:
                continue
        
            # 淘汰旧数据
            while self.current_peds + num_peds > self.max_size and len(self.buffer) > 0:
                index = random.randint(0, len(self.buffer) - 1)
                removed = self.buffer.pop(index)
                removed_hash = self.hash_list.pop(index)
                removed_seq = self.seq_start_ends.pop(index)  # 同步移除对应的seq信息
                self.seen_hashes.remove(removed_hash)
                self.current_peds -= removed[0].shape[1]
        
            if self.current_peds + num_peds <= self.max_size:
                self.buffer.append((obs_segment, pred_segment, logits_segment))
                self.hash_list.append(seq_hash)
                self.seq_start_ends.append([start, end])  # 新增：存储原始[start,end]
                self.seen_hashes.add(seq_hash)
                self.current_peds += num_peds
            else:
                print(f"警告: 无法添加 {num_peds} 行人的序列（超过 max_size={self.max_size})")

    def get_all(self):
        """合并所有非重复数据,并返回seq_start_end"""
        if not self.buffer:
            return None, None, None, None
            
        obs_list, pred_list, logit_list = zip(*self.buffer)
        
        # 重建合并后的seq_start_end
        current_start = 0
        merged_seq_start_end = []
        for obs in obs_list:
            num_peds = obs.shape[1]
            merged_seq_start_end.append([current_start, current_start + num_peds])
            current_start += num_peds
            
        return (
            torch.cat(obs_list, dim=1), 
            torch.cat(pred_list, dim=1),
            torch.cat(logit_list, dim=1),
            torch.tensor(merged_seq_start_end)  # 新增：返回合并后的seq_start_end
        )

    def __len__(self):
        return self.current_peds
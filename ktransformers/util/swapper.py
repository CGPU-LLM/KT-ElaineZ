import os
import torch
from collections import OrderedDict
from typing import Dict, Tuple, Optional
import cpuinfer_ext

class Swapper:
    def __init__(
        self,
        storage_dir: str,
        max_memory_experts: int = 8,
        prefetch_window: int = 2,
        cpu_infer: Optional[cpuinfer_ext.CPUInfer] = None
    ):
        """
        参数：
        storage_dir: 专家权重存储目录
        max_memory_experts: 内存中最多保留的专家数（LRU缓存容量）
        prefetch_window: 预取未来可能使用的专家数量（可用于优化）
        cpu_infer: CPU推理控制器（用于异步IO线程池）
        """
        self.storage_dir = os.path.expanduser(storage_dir)
        self.max_memory = max_memory_experts
        self.prefetch_window = prefetch_window
        self.cpu_infer = cpu_infer or cpuinfer_ext.CPUInfer(1)
        
        # 内存缓存，OrderedDict用于LRU策略
        self.memory_cache = OrderedDict()  # {expert_id: (gate, up, down, metadata)}
        self.on_disk = set()               # 已经落盘的专家ID集合
        self.access_pattern = []           # 访问历史（可用于预取优化）
        self.pending_ops = {}              # 正在进行的异步操作 {expert_id: Future}
        
        os.makedirs(self.storage_dir, exist_ok=True)

    def _async_io_operation(self, fn, *args):
        """封装异步IO操作到CPUInfer线程池，返回Future对象"""
        future = self.cpu_infer.submit(lambda: fn(*args))
        return future

    def _swap_out_lru(self):
        """执行LRU换出策略，将最久未用的专家异步写回磁盘"""
        if not self.memory_cache:
            return

        # 弹出最久未用的专家
        expert_id, data = self.memory_cache.popitem(last=False)
        
        # 异步写回磁盘
        def _save_task():
            file_path = os.path.join(self.storage_dir, f"{expert_id}.pt")
            torch.save({
                'gate': data[0],
                'up': data[1],
                'down': data[2],
                'metadata': data[3]
            }, file_path)
            return True

        self.pending_ops[expert_id] = self._async_io_operation(_save_task)
        self.on_disk.add(expert_id)

    def swap_in(
        self,
        expert_id: str,
        background: bool = False,
        priority: int = 0
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        专家换入操作（支持异步模式）

        参数：
        expert_id: 专家唯一标识
        background: 是否后台异步加载
        priority: 操作优先级（暂未用）

        返回：同步模式返回专家数据，异步模式返回None
        """
        # 等待已有异步操作完成
        if expert_id in self.pending_ops:
            if self.pending_ops[expert_id].done():
                del self.pending_ops[expert_id]
            else:
                if not background:
                    self.pending_ops[expert_id].result()
                else:
                    return None

        # 记录访问历史
        self.access_pattern.append(expert_id)
        if len(self.access_pattern) > 100:
            self.access_pattern.pop(0)
     
        # 已在内存
        if expert_id in self.memory_cache:
            self.memory_cache.move_to_end(expert_id)
            return self.memory_cache[expert_id] if not background else None

        # 内存满则换出
        while len(self.memory_cache) >= self.max_memory - (1 if background else 0):
            self._swap_out_lru()

        # 异步加载
        if background:
            def _load_task():
                file_path = os.path.join(self.storage_dir, f"{expert_id}.pt")
                data = torch.load(file_path, map_location='cpu')
                return (data['gate'], data['up'], data['down'], data['metadata'])
            
            future = self._async_io_operation(_load_task)
            self.pending_ops[expert_id] = future
            return None

        # 同步加载
        file_path = os.path.join(self.storage_dir, f"{expert_id}.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expert {expert_id} not found at {file_path}")
        
        data = torch.load(file_path, map_location='cpu')
        expert_data = (data['gate'], data['up'], data['down'], data['metadata'])
        
        # 放入内存缓存
        self.memory_cache[expert_id] = expert_data
        self.memory_cache.move_to_end(expert_id)
        self.on_disk.discard(expert_id)
        
        return expert_data

    def swap_out(self, expert_id: str, force: bool = False):
        """立即换出指定专家（同步写回磁盘并从内存移除）"""
        if expert_id in self.memory_cache:
            data = self.memory_cache[expert_id]
            file_path = os.path.join(self.storage_dir, f"{expert_id}.pt")
            torch.save({
                'gate': data[0],
                'up': data[1],
                'down': data[2],
                'metadata': data[3]
            }, file_path)
            del self.memory_cache[expert_id]
            self.on_disk.add(expert_id)

    def get_expert(
        self,
        expert_id: str,
        timeout: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取专家数据（如有异步加载则等待，支持超时）

        返回：(gate, up, down)
        """
        if expert_id in self.memory_cache:
            self.memory_cache.move_to_end(expert_id)
            return self.memory_cache[expert_id][:3]
        
        # 等待异步加载完成
        if expert_id in self.pending_ops:
            future = self.pending_ops[expert_id]
            if future.done():
                data = future.result()
                del self.pending_ops[expert_id]
                self.memory_cache[expert_id] = data
                return data[:3]
            elif timeout is not None:
                if future.wait(timeout):
                    data = future.result()
                    del self.pending_ops[expert_id]
                    self.memory_cache[expert_id] = data
                    return data[:3]
                else:
                    raise TimeoutError(f"Loading expert {expert_id} timeout")
        
        # 同步加载
        return self.swap_in(expert_id, background=False)[:3]

    def register_expert(
        self,
        expert_id: str,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        metadata: Dict,
        persist: bool = True
    ):
        """
        注册新专家到管理系统（内存+磁盘）

        参数：
        expert_id: 专家唯一标识
        gate/up/down: 专家权重
        metadata: 附加信息
        persist: 是否立即持久化到磁盘
        """
        # 确保数据在CPU内存并锁页
        gate = gate.cpu().pin_memory()
        up = up.cpu().pin_memory()
        down = down.cpu().pin_memory()
        
        # 放入内存缓存
        self.memory_cache[expert_id] = (gate, up, down, metadata)
        self.memory_cache.move_to_end(expert_id)
        
        # 异步持久化到磁盘
        if persist:
            def _save_task():
                file_path = os.path.join(self.storage_dir, f"{expert_id}.pt")
                torch.save({
                    'gate': gate,
                    'up': up,
                    'down': down,
                    'metadata': metadata
                }, file_path)
                return True
            self.pending_ops[expert_id] = self._async_io_operation(_save_task)
            self.on_disk.add(expert_id)

    def unregister_expert(self, expert_id: str):
        """彻底移除专家（内存+磁盘）"""
        if expert_id in self.memory_cache:
            del self.memory_cache[expert_id]
        if expert_id in self.on_disk:
            file_path = os.path.join(self.storage_dir, f"{expert_id}.pt")
            os.remove(file_path)
            self.on_disk.remove(expert_id)

    def flush(self):
        """确保所有异步操作完成（阻塞）"""
        for expert_id in list(self.pending_ops.keys()):
            if self.pending_ops[expert_id].done():
                del self.pending_ops[expert_id]
            else:
                self.pending_ops[expert_id].result()
                del self.pending_ops[expert_id]

    def __del__(self):
        """析构时确保资源释放"""
        self.flush()
        if hasattr(cpuinfer_ext, 'release_resources'):
            cpuinfer_ext.release_resources()
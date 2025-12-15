
# agents/torch_replay.py
from __future__ import annotations
from typing import Tuple
import torch


class TorchReplayBuffer:
    """
    Fast replay buffer implemented as preallocated tensors (ring buffer).

    Stores on CPU (optionally pinned) and samples with torch.randint,
    then transfers batches to the desired device with non_blocking=True.

    Removes Python deque/random.sample overhead and avoids repeatedly building
    tensors from Python lists.
    """

    def __init__(self, capacity: int, obs_dim: int, pin_memory: bool = True):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.ptr = 0
        self.size = 0

        kwargs = {}
        if pin_memory and torch.cuda.is_available():
            kwargs["pin_memory"] = True

        self.s = torch.empty((self.capacity, self.obs_dim), dtype=torch.float32, device="cpu", **kwargs)
        self.a = torch.empty((self.capacity,), dtype=torch.int64, device="cpu", **kwargs)
        self.r = torch.empty((self.capacity,), dtype=torch.float32, device="cpu", **kwargs)
        self.s2 = torch.empty((self.capacity, self.obs_dim), dtype=torch.float32, device="cpu", **kwargs)
        self.d = torch.empty((self.capacity,), dtype=torch.float32, device="cpu", **kwargs)

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def push_batch(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        s2: torch.Tensor,
        d: torch.Tensor,
    ) -> None:
        if s.ndim != 2:
            raise ValueError("s must be [B, obs_dim]")
        B = int(s.shape[0])
        if B == 0:
            return

        if B > self.capacity:
            s = s[-self.capacity:]
            a = a[-self.capacity:]
            r = r[-self.capacity:]
            s2 = s2[-self.capacity:]
            d = d[-self.capacity:]
            B = self.capacity

        end = self.ptr + B
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.s[sl].copy_(s)
            self.a[sl].copy_(a)
            self.r[sl].copy_(r)
            self.s2[sl].copy_(s2)
            self.d[sl].copy_(d)
        else:
            k1 = self.capacity - self.ptr
            k2 = B - k1
            sl1 = slice(self.ptr, self.capacity)
            sl2 = slice(0, k2)

            self.s[sl1].copy_(s[:k1])
            self.a[sl1].copy_(a[:k1])
            self.r[sl1].copy_(r[:k1])
            self.s2[sl1].copy_(s2[:k1])
            self.d[sl1].copy_(d[:k1])

            self.s[sl2].copy_(s[k1:])
            self.a[sl2].copy_(a[k1:])
            self.r[sl2].copy_(r[k1:])
            self.s2[sl2].copy_(s2[k1:])
            self.d[sl2].copy_(d[k1:])

        self.ptr = end % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("Not enough samples in replay.")
        idx = torch.randint(0, self.size, (batch_size,), device="cpu")
        s = self.s[idx].to(device, non_blocking=True)
        a = self.a[idx].to(device, non_blocking=True)
        r = self.r[idx].to(device, non_blocking=True)
        s2 = self.s2[idx].to(device, non_blocking=True)
        d = self.d[idx].to(device, non_blocking=True)
        return s, a, r, s2, d

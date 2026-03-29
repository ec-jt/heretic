# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
from dataclasses import dataclass
from contextlib import suppress

import torch
import torch.distributed as dist


@dataclass
class DistributedState:
    enabled: bool
    backend: str
    world_size: int
    rank: int
    local_rank: int
    master_addr: str
    master_port: int

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def get_distributed_state(enabled: bool, backend: str) -> DistributedState:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))

    # Auto-enable distributed mode when launched with torchrun.
    active = enabled or world_size > 1

    return DistributedState(
        enabled=active,
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
    )


def init_distributed(state: DistributedState):
    if not state.enabled:
        return

    if state.world_size <= 1:
        return

    if dist.is_available() and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(state.local_rank)

        dist.init_process_group(
            backend=state.backend,
            rank=state.rank,
            world_size=state.world_size,
            init_method=f"tcp://{state.master_addr}:{state.master_port}",
        )


def barrier(state: DistributedState):
    if state.enabled and state.world_size > 1 and dist.is_initialized():
        dist.barrier()


def destroy_distributed():
    with suppress(Exception):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Optional

from vllm import envs
from vllm.distributed.af_transfer.af_connector import AFConnectorBase
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_world_group

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_AF_CONNECTOR: Optional[AFConnectorBase] = None


def has_af_connector() -> bool:
    return _AF_CONNECTOR is not None


def get_af_connector() -> AFConnectorBase:
    assert has_af_connector(), (
        "disaggregated KV cache transfer parallel group is not initialized"
    )
    return _AF_CONNECTOR


def ensure_kv_transfer_initialized(vllm_config: "VllmConfig") -> None:
    """
    Initialize KV cache transfer parallel group.
    """

    global _AF_CONNECTOR

    if vllm_config.kv_transfer_config is None:
        return

    if (
        vllm_config.kv_transfer_config.is_kv_transfer_instance
        and _AF_CONNECTOR is None
    ):
        if envs.VLLM_USE_V1:
            _AF_CONNECTOR = KVConnectorFactory.create_connector_v1(
                config=vllm_config, role=KVConnectorRole.WORKER
            )
        else:
            _AF_CONNECTOR = KVConnectorFactory.create_connector_v0(
                rank=get_world_group().rank,
                local_rank=get_world_group().local_rank,
                config=vllm_config,
            )

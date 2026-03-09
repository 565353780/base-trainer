from torch import nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh


def default_fsdp_shard_fn(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
) -> None:
    """Apply fully_shard bottom-up to each sub-module before the root.

    For a Transformer-like model you would iterate over layers:
        for layer in model.layers:
            fully_shard(layer, mesh=device_mesh, mp_policy=mp_policy)

    The root module is sharded automatically by BaseTrainer after this
    function returns, so do NOT call fully_shard on the root here.
    """
    for child in model.children():
        fully_shard(child, mesh=device_mesh, mp_policy=mp_policy)
    return

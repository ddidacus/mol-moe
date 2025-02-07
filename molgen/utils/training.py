from .compute import is_master_rank
import re

def set_trainable_routers_only(policy):
    """
        Set only the routers as trainable in the MoE model
        Arguments:
            policy(object): HF loaded policy model
            accelerator(object): HF accelerator object
    """
    # Set only the routers as trainable in the MoE model
    pattern = re.compile(r'.*block_sparse_moe\.gate.*')
    for name, module in policy.named_modules():
        if pattern.match(name):
            for param in module.parameters():
                param.requires_grad = True
            print(f"Set requires_grad=True for matching layer: {name}")
        else:
            for param in module.parameters():
                param.requires_grad = False
    
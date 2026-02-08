import torch
from typing import List, Dict


def get_parent_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """
    Get the parent module of a given module by name.
    
    Args:
        model: The model
        name: Full name of the module (e.g., 'model.layers.0.self_attn.o_proj')
    
    Returns:
        Parent module
    """
    names = name.split('.')
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent


def inject_adapters_vlm(
    model: torch.nn.Module,
    adapter_cls: type,
    base_adapter_args: dict,
    layers_config: List[Dict[str, str]]
) -> torch.nn.Module:
    """
    Inject DCT adapters into VLM model at specified layers.
    
    Args:
        model: The VLM model (Idefics2, LLaVA, etc.)
        adapter_cls: The adapter class (DCTAdapter)
        base_adapter_args: Base arguments for adapter initialization
        layers_config: List of layer configurations with 'name' key for pattern matching
    
    Returns:
        Model with injected adapters
    """
    print(f"Starting adapter injection with {adapter_cls.__name__}...")
    
    injection_count = 0
    
    for name, module in model.named_modules():
        for layer_conf in layers_config:
            # Check if the current module's name matches the configuration pattern
            if layer_conf['name'] in name:
                print(f"Matched target layer for injection: {name}")
                try:
                    parent = get_parent_module(model, name)
                    original_module = getattr(parent, name.split('.')[-1])
                    
                    current_adapter_args = base_adapter_args.copy()

                    # Dynamically set in_features based on the original module's output dimension
                    if hasattr(original_module, 'out_features') and isinstance(getattr(original_module, 'out_features'), int):
                        actual_in_features = original_module.out_features
                        print(f"Dynamically setting adapter 'in_features' for {name} to {actual_in_features} (from out_features).")
                        current_adapter_args['in_features'] = actual_in_features
                    elif hasattr(original_module, 'hidden_size') and isinstance(getattr(original_module, 'hidden_size'), int):
                        actual_in_features = original_module.hidden_size
                        print(f"Dynamically setting adapter 'in_features' for {name} to {actual_in_features} (from hidden_size).")
                        current_adapter_args['in_features'] = actual_in_features
                    else:
                        print(f"Original module {name} (type: {type(original_module)}) does not have suitable 'out_features' or 'hidden_size'. "
                              f"Using 'in_features' from base_adapter_args: {current_adapter_args.get('in_features')}. "
                              f"This might lead to errors if incorrect for this layer.")

                    adapter_instance = adapter_cls(**current_adapter_args)
                    
                    # Wrap original module + adapter in Sequential
                    setattr(parent, name.split('.')[-1], torch.nn.Sequential(original_module, adapter_instance))
                    
                    print(f"Successfully injected adapter after {name} with args: {current_adapter_args}")
                    injection_count += 1
                    
                except Exception as e:
                    print(f"Failed to inject adapter into {name}: {e}")
    
    print(f"Adapter injection complete. Total adapters injected: {injection_count}")
    return model


def freeze_model_except_adapters(model: torch.nn.Module):
    """
    Freeze all model parameters except those in adapter layers.
    
    Args:
        model: The model with injected adapters
    """
    for name, param in model.named_parameters():
        if "Sequential" in name or "adapter" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
        # print(name)
    
    # Print statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params_m = trainable_params / 1_000_000
    all_params_m = all_params / 1_000_000
    
    print(f"\n{'='*60}")
    print(f"Trainable params: {trainable_params:,} ({trainable_params_m:.2f}M) || All params: {all_params:,} ({all_params_m:.2f}M)")
    print(f"Trainable%: {trainable_params/all_params*100:.4f}%")
    print(f"{'='*60}\n")



def print_trainable_parameters(model: torch.nn.Module):
    """
    Print all trainable parameters in the model.
    
    Args:
        model: The model to inspect
    """
    print("\nTrainable parameters:")
    print("-" * 80)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✓ {name:60} | shape: {str(list(param.shape)):20} | dtype: {param.dtype}")
    print("-" * 80)


from __future__ import annotations
import hashlib
from pathlib import Path

def require_torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("Install model extras: pip install -e '.[model]'") from e
    return torch

def load_student(model_id, revision=None, device_map="auto", dtype="bfloat16",
                 trust_remote_code=False, load_in_4bit=False):
    torch = require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}
    if dtype not in dtypes:
        raise ValueError(f"Unsupported dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                              trust_remote_code=trust_remote_code)
    kwargs = dict(revision=revision, device_map=device_map,
                  torch_dtype=dtypes[dtype], trust_remote_code=trust_remote_code)
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer

def discover_linear_modules(model):
    """Return named torch.nn.Linear modules and inferred transformer block index."""
    torch = require_torch()
    found = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            block = None
            parts = name.split(".")
            for i, part in enumerate(parts[:-1]):
                if part in {"layers", "blocks", "h"}:
                    try: block = int(parts[i+1])
                    except ValueError: pass
            found.append({"name": name, "block_index": block,
                          "d_in": module.in_features, "d_out": module.out_features,
                          "module": module})
    return found

class ActivationCapture:
    """Context manager capturing outputs from explicitly selected modules."""
    def __init__(self, model, module_names):
        self.model, self.module_names = model, list(module_names)
        self.handles, self.outputs = [], {}
    def __enter__(self):
        modules = dict(self.model.named_modules())
        missing = set(self.module_names) - set(modules)
        if missing:
            raise KeyError(f"Unknown modules: {sorted(missing)}")
        for name in self.module_names:
            def hook(_module, _inputs, output, key=name):
                value = output[0] if isinstance(output, tuple) else output
                self.outputs.setdefault(key, []).append(value.detach().float().cpu())
            self.handles.append(modules[name].register_forward_hook(hook))
        return self
    def __exit__(self, exc_type, exc, tb):
        for h in self.handles: h.remove()
        self.handles.clear()

def tensor_hash(tensor):
    import torch
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()

def attach_lora(model, module_name, a, b, alpha, dose=1.0):
    """Attach a reversible forward hook implementing dose*(alpha/r)*B(A(x))."""
    torch = require_torch()
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(module_name)
    a_t = torch.as_tensor(a, dtype=next(modules[module_name].parameters()).dtype,
                          device=next(modules[module_name].parameters()).device)
    b_t = torch.as_tensor(b, dtype=a_t.dtype, device=a_t.device)
    rank = int(a_t.shape[0])
    scale = float(dose) * float(alpha) / rank
    def hook(_module, inputs, output):
        x = inputs[0]
        low = torch.nn.functional.linear(x, a_t)
        delta = torch.nn.functional.linear(low, b_t) * scale
        return output + delta
    handle = modules[module_name].register_forward_hook(hook)
    return handle

def unload_adapter(handles):
    for handle in handles:
        handle.remove()

def verify_base_unchanged(before_hashes, model):
    modules = dict(model.named_modules())
    after = {}
    for name, module in modules.items():
        if hasattr(module, "weight") and module.weight is not None:
            after[name] = tensor_hash(module.weight)
    changed = [name for name, digest in before_hashes.items()
               if after.get(name) != digest]
    return {"unchanged": not changed, "changed_modules": changed}

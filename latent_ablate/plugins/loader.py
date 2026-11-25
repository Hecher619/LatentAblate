import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger("LatentAblate.PluginLoader")


def load_python_plugin(plugin_path: str, expected_attr: str | None = None):
    """
    Dynamically loads a Python module from `plugin_path` and resolves a plugin entry
    object/callable (`Plugin`, `Scorer`, `Modifier`, or `expected_attr`).
    """
    path = Path(plugin_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Plugin file not found: {plugin_path}")

    module_name = f"latent_ablate_ext_{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for plugin: {plugin_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidates = []
    if expected_attr:
        candidates.append(expected_attr)
    candidates.extend(["Plugin", "Modifier", "Scorer", "get_plugin", "run_surgery", "get_score"])

    for attr in candidates:
        if hasattr(module, attr):
            obj = getattr(module, attr)
            return obj() if isinstance(obj, type) else obj

    return module


class PluginLoader:
    """Compatibility loader wrapper around load_python_plugin."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def load(self, path, *args, **kwargs):
        return load_python_plugin(path, *args, **kwargs)

    def __call__(self, path, *args, **kwargs):
        return self.load(path, *args, **kwargs)

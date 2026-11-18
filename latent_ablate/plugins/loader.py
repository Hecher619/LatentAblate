import importlib.util
from pathlib import Path
import logging

logger = logging.getLogger("LatentAblate.PluginLoader")

class PluginLoader:
    @staticmethod
    def load_plugin_from_file(file_path: str, base_class):
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {path}")

        logger.info(f"Dynamically loading plugin from {path}...")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, base_class) and attr is not base_class:
                logger.info(f"Successfully loaded plugin class: {attr.__name__}")
                return attr

        raise ValueError(f"No valid subclass of {base_class.__name__} found in {path}")

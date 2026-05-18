"""pytest config — makes `steps.*` and `tools.*` importable as modules."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_step(filename: str):
    """Load a `steps/NN_<name>.py` module by file (since `NN_` prefix isn't a valid identifier).

    Registers the loaded module in sys.modules so @dataclass and other things that
    walk sys.modules during class construction can find it.
    """
    mod_name = "_step_" + filename.replace(".py", "").replace(".", "_")
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = ROOT / "steps" / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

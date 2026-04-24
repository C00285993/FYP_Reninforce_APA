"""Shared model loading utility — auto-detects DQN vs PPO."""
import contextlib
import io
import os
import threading

from stable_baselines3 import DQN, PPO

# Process-wide lock so only one thread redirects stdout at a time
_load_lock = threading.Lock()


def load_model(model_path: str, env, verbose: int = 0):
    """Try loading as DQN, then PPO. Returns (model, algo_name).

    verbose=0 suppresses the 'Wrapping the env in a DummyVecEnv' message
    that SB3 prints to stdout during model loading.
    Thread-safe: uses a lock to serialize stdout redirection.
    """
    with _load_lock:
        if verbose == 0:
            redirect = contextlib.redirect_stdout(io.StringIO())
        else:
            redirect = contextlib.nullcontext()
        for cls, name in [(DQN, "DQN"), (PPO, "PPO")]:
            try:
                with redirect:
                    model = cls.load(model_path, env=env)
                model.verbose = 0
                return model, name
            except Exception:
                # Reset redirect for next attempt (StringIO may be consumed)
                if verbose == 0:
                    redirect = contextlib.redirect_stdout(io.StringIO())
                continue
    raise ValueError(f"Could not load model from {model_path}")

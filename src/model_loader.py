# Deprecated: YOLO is no longer used.
# This file is kept for compatibility but should be removed in future cleanup.

def load_model(model_basename, task="detect"):
    print(f"Warning: load_model({model_basename}) was called, but YOLO is deprecated.")
    return None
"""Run frozen Phase 1 inference over calibration/test_id/test_ood and save raw predictions."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase2.inference import run_frozen_inference, save_predictions  # noqa: E402

if __name__ == "__main__":
    for split in ["calibration", "test_id", "test_ood"]:
        t0 = time.time()
        df = run_frozen_inference(split)
        path = save_predictions(df, split)
        print(f"[{split}] rows={len(df)} graphs={df['graph_id'].nunique()} "
              f"saved={path} elapsed={time.time()-t0:.1f}s")

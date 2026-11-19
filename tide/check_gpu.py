from __future__ import annotations

from sdge_microgrid.device import cuda_summary, require_cuda_for_tide

if __name__ == "__main__":
    print("device:", require_cuda_for_tide())
    print("summary:", cuda_summary())

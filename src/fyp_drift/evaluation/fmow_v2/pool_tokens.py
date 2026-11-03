"""Average 2x2 patch tokens (20x20 -> 10x10) so the head-training caches fit in RAM."""

from pathlib import Path

import numpy as np


def main():
    src, dst = (Path("/mnt/sdd/FYP/cache/adapter_tokens_v2"), Path("/mnt/sdd/FYP/cache/adapter_tokens_v2p"))
    dst.mkdir(exist_ok=True)
    for f in sorted(src.glob("*.npy")):
        out = dst / f.name
        if out.with_suffix(".done").exists():
            continue
        x = np.load(f, mmap_mode="r")
        n, p, d = x.shape
        g = int(round(p**0.5))
        assert g * g == p and g % 2 == 0
        y = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n, (g // 2) ** 2, d))
        for s in range(0, n, 256):
            b = np.asarray(x[s : s + 256], dtype=np.float32).reshape(-1, g // 2, 2, g // 2, 2, d).mean((2, 4))
            y[s : s + 256] = b.reshape(len(b), -1, d).astype(np.float16)
        y.flush()
        out.with_suffix(".done").write_text("ok")
        print("pooled", f.name, x.shape, "->", y.shape, flush=True)


if __name__ == "__main__":
    main()

import os
from pathlib import Path
import multiprocessing as mp
from litdata import optimize

CLASSES = [f"class_{i:02d}" for i in range(11)]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
CHUNK_SIZE = int(os.getenv("LITDATA_CHUNK_SIZE", "1024"))

def pack_image(sample_path: str):
    label_name = Path(sample_path).parent.name
    if label_name not in CLASSES:
        raise ValueError(f"Unexpected class folder: {label_name} for {sample_path}")

    label = CLASSES.index(label_name)
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    return {"image": img_bytes, "label": label, "path": str(sample_path)}

def list_images(split_dir: str):
    p = Path(split_dir)
    files = [str(x) for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def main():
    food11_local_dir = os.environ["FOOD11_LOCAL_DIR"]  # contains training/validation/evaluation
    local_out_root = os.getenv("LITDATA_LOCAL_OUT", "/mnt/local/litdata_food11")

    for split in ["training", "validation", "evaluation"]:
        split_dir = os.path.join(food11_local_dir, split)
        out_dir = os.path.join(local_out_root, split)

        inputs = list_images(split_dir)
        print(f"[litdata] split={split} files={len(inputs)} -> local_out={out_dir}", flush=True)
        if len(inputs) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

        optimize(
            inputs=inputs,
            output_dir=out_dir,
            fn=pack_image,
            chunk_size=CHUNK_SIZE,
        )

    print("[done] local shards written to:", local_out_root, flush=True)

if __name__ == "__main__":
    # python3.12 + spawn needs this guard
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
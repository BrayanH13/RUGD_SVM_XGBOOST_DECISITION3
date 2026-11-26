import argparse
import shutil
from pathlib import Path

import cv2


def frames(scene_dir):
    return sorted(p for p in Path(scene_dir).glob("*.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--video-scene", default="trail-5")
    ap.add_argument("--video-frames", type=int, default=150)
    ap.add_argument("--fps", type=float, default=6)
    args = ap.parse_args()

    root = Path(args.frames_dir)
    out = Path(args.out)
    (out / "imagenes").mkdir(parents=True, exist_ok=True)

    picks = [("trail-4", 0.3), ("trail-11", 0.6), ("trail-14", 0.5),
             ("creek", 0.4), ("park-2", 0.5), ("village", 0.5)]
    for scene, pos in picks:
        fs = frames(root / scene)
        if not fs:
            print(f"  (sin frames en {scene})")
            continue
        src = fs[int(pos * (len(fs) - 1))]
        tag = "visto" if scene.startswith("trail") else "no_visto"
        dst = out / "imagenes" / f"{tag}_{src.name}"
        shutil.copy(src, dst)
        print(f"  imagen: {dst.name}")

    fs = frames(root / args.video_scene)
    if fs:
        start = max(0, (len(fs) - args.video_frames) // 2)
        sel = fs[start:start + args.video_frames]
        h, w = cv2.imread(str(sel[0])).shape[:2]
        path = out / f"recorrido_{args.video_scene}.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for f in sel:
            vw.write(cv2.imread(str(f)))
        vw.release()
        print(f"  video: {path.name} ({len(sel)} frames, {len(sel) / args.fps:.0f} s)")


if __name__ == "__main__":
    main()

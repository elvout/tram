import sys
from pathlib import Path

import estimate_camera
import estimate_humans
import visualize_tram

if __name__ == "__main__":
    SUCCESS_FLAG_DIR = Path("logs/success")
    FAIL_FLAG_DIR = Path("logs/fail")

    SUCCESS_FLAG_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_FLAG_DIR.mkdir(parents=True, exist_ok=True)

    for video_str in sys.argv[1:]:
        video = Path(video_str)
        if (SUCCESS_FLAG_DIR / video.name).exists():
            print(
                "\n".join(
                    (
                        f"Skipping {video.name}: already ran inference successfully",
                        f"Delete {SUCCESS_FLAG_DIR}/{video.name} if you wan to re-run inference",
                    )
                )
            )
            continue

        status_flag = FAIL_FLAG_DIR / video.name
        status_flag.symlink_to(video)

        try:
            estimate_camera.main(str(video), static_camera=True, visualize_mask=False)
            estimate_humans.main(str(video))
            # visualize_tram.main(str(video))
        except:  # noqa: E722
            continue

        status_flag.rename(SUCCESS_FLAG_DIR / video.name)

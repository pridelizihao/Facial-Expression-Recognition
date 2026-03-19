from ultralytics.utils.plotting import plot_results

plot_results(
    file="runs/classify/archive_yolo26x_cls/results.csv",
    classify=True
)

import swanlab

swanlab.log({
    "training_curve": swanlab.Image(
        "runs/classify/archive_yolo26x_cls/results.png"
    )
})
## Installation


```bash
git clone https://github.com/naurril/ad-metrics.git
cd ad-metrics

# install env （optional）
python3 -m venv .venv
.venv/bin/activate

# install dependencies
pip install -r requirements.txt

# install ad-metrics
pip install -e .
```


## Test

```
pytest tests
```


## Count all metrics calculation APIs

``` bash
grep 'def calculate_' admetrics -r |wc -l

```

## Quick Start

### Detection Evaluation

```python
import numpy as np
from admetrics.detection.iou import calculate_iou_3d
from admetrics.detection.ap import calculate_ap
from admetrics.detection.nds import calculate_nds

# Minimal example inputs
box1 = np.array([0, 0, 0, 4, 2, 1.5, 0])  # [x, y, z, w, h, l, yaw]
box2 = np.array([1, 0, 0, 4, 2, 1.5, 0])

iou = calculate_iou_3d(box1, box2)
print('IoU:', iou)

predictions = [{'box': box1, 'score': 0.9, 'class': 'car'}]
ground_truth = [{'box': box2, 'class': 'car'}]

ap_result = calculate_ap(predictions, ground_truth, iou_threshold=0.7)
print(f"AP@0.7: {ap_result['ap']:.4f}")

nds = calculate_nds(predictions, ground_truth, class_names=['car'])
print(f"NDS: {nds:.4f}")
```

### Tracking Evaluation

```python
import numpy as np
from admetrics.tracking.tracking import (
    calculate_multi_frame_mota,
    calculate_hota,
    calculate_amota,
    calculate_tid_lgd,
    calculate_id_f1,
)

# Minimal frame-based inputs: mapping frame_idx -> list of detections/gt
frame_preds = {0: [{'box': np.array([0, 0, 0, 1, 1, 1, 0]), 'score': 0.9, 'class': 'car'}]}
frame_gts = {0: [{'box': np.array([0, 0, 0, 1, 1, 1, 0]), 'class': 'car'}]}

results = calculate_multi_frame_mota(frame_preds, frame_gts)
print(f"MOTA: {results['mota']:.4f}")
print(f"ID Switches: {results.get('num_switches', 0)}")

hota_results = calculate_hota(frame_preds, frame_gts)
print(f"HOTA: {hota_results['hota']:.4f}")

amota_results = calculate_amota(frame_preds, frame_gts, recall_thresholds=[0.2, 0.4])
print(f"AMOTA: {amota_results['amota']:.4f}")

tid_lgd = calculate_tid_lgd(frame_preds, frame_gts)
print(f"TID: {tid_lgd.get('tid', 0):.2f}, LGD: {tid_lgd.get('lgd', 0):.2f}")

idf1_results = calculate_id_f1(frame_preds, frame_gts)
print(f"IDF1: {idf1_results['idf1']:.4f}")
```

### Trajectory Prediction

```python
import numpy as np
from admetrics.prediction.trajectory import calculate_ade, calculate_fde, calculate_multimodal_ade

# predicted and gt as (T, 2) arrays
predicted_traj = np.array([[0., 0.], [1., 0.], [2., 0.]])
ground_truth_traj = np.array([[0., 0.], [1., 0.1], [2., 0.2]])

ade = calculate_ade(predicted_traj, ground_truth_traj)
fde = calculate_fde(predicted_traj, ground_truth_traj)
print(f"ADE: {ade:.3f}, FDE: {fde:.3f}")

# multimodal example: list of K mode trajectories
modes = [predicted_traj, predicted_traj + 0.1]
mm = calculate_multimodal_ade(modes, ground_truth_traj)
print('Multimodal keys:', list(mm.keys()))
```

### Localization Evaluation

```python
import numpy as np
from admetrics.localization.localization import calculate_localization_metrics

# simple 2D trajectory mapped as (N, 2) for this example
pred = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])
gt = pred + 0.01
timestamps = np.linspace(0, 1, 10)

metrics = calculate_localization_metrics(pred, gt, timestamps=timestamps, lane_width=3.5, align=False)
print('ATE mean:', metrics['ate_mean'])
print('Lateral mean:', metrics['lateral_mean'])
```

### Occupancy Prediction

```python
import numpy as np
from admetrics.occupancy.occupancy import calculate_mean_iou, calculate_scene_completion

pred = np.zeros((10, 10, 1), dtype=int)
gt = np.zeros_like(pred)
miou = calculate_mean_iou(pred, gt, num_classes=1)
print('mIoU keys:', list(miou.keys()))

sc_result = calculate_scene_completion(pred, gt)
print('Scene completion keys:', list(sc_result.keys()))
```

### Vector Map Detection

```python
import numpy as np
from admetrics.vectormap.vectormap import calculate_chamfer_distance_polyline, calculate_topology_metrics

pred_poly = np.array([[0, 0], [1, 0]])
gt_poly = np.array([[0, 0], [1, 0]])
cd = calculate_chamfer_distance_polyline(pred_poly, gt_poly)
print('Chamfer keys:', list(cd.keys()))

# topology requires lane_matches argument; provide empty list for minimal example
pred_graph = {'nodes': [], 'edges': []}
gt_graph = {'nodes': [], 'edges': []}
lane_matches = []
topo = calculate_topology_metrics(pred_graph, gt_graph, lane_matches)
print('Topology keys:', list(topo.keys()))
```

### Planning Evaluation

```python
import numpy as np
from admetrics.planning.planning import calculate_driving_score, calculate_collision_rate

pred_traj = np.array([[0, 0], [1, 0], [2, 0]])
expert = pred_traj.copy()
obstacles = [np.array([[5, 5]])]
timestamps = np.linspace(0, 1, len(pred_traj))

score = calculate_driving_score(pred_traj, expert, obstacles, pred_traj, timestamps)
print('Driving score keys:', list(score.keys()))

collision = calculate_collision_rate(pred_traj, obstacles)
print('Collision keys:', list(collision.keys()))
```

### Simulation Quality Evaluation

```python
import numpy as np
from admetrics.simulation.camera_metrics import calculate_camera_image_quality
from admetrics.simulation.lidar_metrics import calculate_lidar_point_cloud_quality
from admetrics.simulation.sim2real_metrics import calculate_perception_sim2real_gap
from admetrics.simulation.weather_metrics import calculate_weather_simulation_quality

# small dummy images
sim_images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
real_images = np.zeros_like(sim_images)
cam_metrics = calculate_camera_image_quality(sim_images, real_images)
print('camera keys:', list(cam_metrics.keys()))

sim_pts = np.random.rand(10, 3)
real_pts = np.random.rand(8, 3)
lidar_metrics = calculate_lidar_point_cloud_quality(sim_pts, real_pts)
print('lidar keys:', list(lidar_metrics.keys()))

sim_det = [{'box': np.array([0, 0, 0, 1, 1, 1, 0]), 'score': 0.9, 'class': 'car'}]
real_det = sim_det
gap = calculate_perception_sim2real_gap(sim_det, real_det)
print('sim2real keys:', list(gap.keys()))

weather = calculate_weather_simulation_quality({'rain': sim_images}, {'rain': real_images}, weather_type='rain')
print('weather keys:', list(weather.keys()))
```

### Utilities (matching, NMS, transforms, visualization)

```python
import numpy as np
from admetrics.utils.matching import greedy_matching
from admetrics.utils.nms import nms_bev
from admetrics.utils.transforms import convert_box_format
from admetrics.utils.visualization import plot_boxes_bev

preds = [{'box': np.array([0, 0, 0, 1, 1, 1, 0]), 'score': 0.9}]
gts = [{'box': np.array([0, 0, 0, 1, 1, 1, 0])}]
matches = greedy_matching(preds, gts)
print('matches:', matches)

keep = nms_bev([{'box': np.array([0, 0, 0, 1, 1, 1, 0]), 'score': 0.9}])
print('nms kept:', keep)

box = np.array([0, 0, 0, 1, 1, 1, 0])
cf = convert_box_format(box, src_format='xyzwhlr', dst_format='xyzwhlr')
print('convert box ok')
```
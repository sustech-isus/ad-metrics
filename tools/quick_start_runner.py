import numpy as np
import traceback

print("Quick start runner: executing example snippets with dummy data\n")

errors = []

# Helper to run blocks
def run_block(name, fn):
    print(f"--- Running: {name} ---")
    try:
        fn()
        print(f"{name}: OK\n")
    except Exception as e:
        print(f"{name}: ERROR -> {e}\n")
        traceback.print_exc()
        errors.append((name, e))


# 1) Detection Evaluation
def block_detection():
    from admetrics.detection import calculate_iou_3d
    from admetrics.detection.ap import calculate_ap
    from admetrics.detection.nds import calculate_nds

    box1 = np.array([0, 0, 0, 4, 2, 1.5, 0])
    box2 = np.array([1, 0, 0, 4, 2, 1.5, 0])
    iou = calculate_iou_3d(box1, box2)
    print('iou=', iou)

    # minimal predictions/ground_truth format expected by calculate_ap
    predictions = [{'box': box1, 'score': 0.9, 'class': 'car'}]
    ground_truth = [{'box': box2, 'class': 'car'}]
    ap_res = calculate_ap(predictions, ground_truth, iou_threshold=0.5)
    print('ap keys:', list(ap_res.keys()))

    # `calculate_nds` does not accept `class_weights` kwarg in this implementation
    # calculate_nds expects a list of class names
    nds_val = calculate_nds(predictions, ground_truth, class_names=['car'])
    # calculate_nds returns a float (overall score) in this implementation
    if hasattr(nds_val, 'keys'):
        print('nds keys:', list(nds_val.keys()))
    else:
        print('nds value:', nds_val)

# 2) Tracking Evaluation
def block_tracking():
    from admetrics.tracking.tracking import calculate_multi_frame_mota, calculate_hota, calculate_amota, calculate_tid_lgd, calculate_id_f1
    # create minimal frame dicts
    frame_preds = {0: [{'box': np.array([0,0,0,1,1,1,0]), 'score':0.9, 'class': 'car'}]}
    frame_gts = {0: [{'box': np.array([0,0,0,1,1,1,0]), 'class': 'car'}]}
    res = calculate_multi_frame_mota(frame_preds, frame_gts)
    print('mota keys:', list(res.keys()))
    hota = calculate_hota(frame_preds, frame_gts)
    print('hota keys:', list(hota.keys()))
    amota = calculate_amota(frame_preds, frame_gts, recall_thresholds=[0.2,0.4])
    print('amota keys:', list(amota.keys()))
    tid = calculate_tid_lgd(frame_preds, frame_gts)
    print('tid keys:', list(tid.keys()))
    idf1 = calculate_id_f1(frame_preds, frame_gts)
    print('idf1 keys:', list(idf1.keys()))

# 3) Trajectory Prediction
def block_prediction():
    from admetrics.prediction.trajectory import calculate_ade, calculate_fde, calculate_multimodal_ade
    pred = np.array([[0.,0.],[1.,0.],[2.,0.]])
    gt = np.array([[0.,0.],[1.,0.1],[2.,0.2]])
    ade = calculate_ade(pred, gt)
    fde = calculate_fde(pred, gt)
    print('ade,fde:', ade, fde)
    # multimodal: provide K modes
    modes = [pred, pred+0.1]
    mm = calculate_multimodal_ade(modes, gt)
    print('multimodal keys:', list(mm.keys()))

# 4) Localization Evaluation
def block_localization():
    from admetrics.localization.localization import calculate_localization_metrics
    # create simple poses as (x,y) trajectory for wrapper
    pred = np.column_stack([np.linspace(0,1,10), np.zeros(10)])
    gt = pred + 0.01
    metrics = calculate_localization_metrics(pred, gt, timestamps=np.linspace(0,1,10), lane_width=3.5, align=False)
    print('localization keys:', list(metrics.keys()))

# 5) Occupancy Prediction
def block_occupancy():
    from admetrics.occupancy.occupancy import calculate_mean_iou, calculate_scene_completion
    pred = np.zeros((10,10,1), dtype=int)
    gt = np.zeros_like(pred)
    miou = calculate_mean_iou(pred, gt, num_classes=1)
    print('miou keys:', list(miou.keys()))
    sc = calculate_scene_completion(pred, gt)
    print('scene completion keys:', list(sc.keys()))

# 6) Vector Map Detection
def block_vectormap():
    from admetrics.vectormap.vectormap import calculate_chamfer_distance_polyline, calculate_topology_metrics
    pred = [np.array([[0,0],[1,0]])]
    gt = [np.array([[0,0],[1,0]])]
    cd = calculate_chamfer_distance_polyline(pred[0], gt[0])
    print('chamfer keys:', list(cd.keys()))
    # minimal graphs for topology
    pred_graph = {'nodes': [], 'edges': []}
    gt_graph = {'nodes': [], 'edges': []}
    # provide minimal lane_matches (empty) as the implementation requires it
    lane_matches = []
    topo = calculate_topology_metrics(pred_graph, gt_graph, lane_matches)
    print('topo keys:', list(topo.keys()))

# 7) Planning Evaluation
def block_planning():
    from admetrics.planning.planning import calculate_driving_score, calculate_collision_rate
    pred_traj = np.array([[0,0],[1,0],[2,0]])
    expert = pred_traj.copy()
    obstacles = [np.array([[5,5]])]
    timestamps = np.linspace(0,1,3)
    score = calculate_driving_score(pred_traj, expert, obstacles, pred_traj, timestamps)
    print('driving score keys:', list(score.keys()))
    coll = calculate_collision_rate(pred_traj, obstacles)
    print('collision keys:', list(coll.keys()))

# 8) Simulation Quality
def block_simulation():
    from admetrics.simulation.camera_metrics import calculate_camera_image_quality
    from admetrics.simulation.lidar_metrics import calculate_lidar_point_cloud_quality
    from admetrics.simulation.sim2real_metrics import calculate_perception_sim2real_gap
    from admetrics.simulation.weather_metrics import calculate_weather_simulation_quality
    # dummy images: small arrays
    sim_images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    real_images = np.zeros_like(sim_images)
    cam = calculate_camera_image_quality(sim_images, real_images)
    print('camera keys:', list(cam.keys()))
    sim_pts = np.random.rand(10,3)
    real_pts = np.random.rand(8,3)
    lidar = calculate_lidar_point_cloud_quality(sim_pts, real_pts)
    print('lidar keys:', list(lidar.keys()))
    sim_det = [{'box': np.array([0,0,0,1,1,1,0]), 'score':0.9, 'class':'car'}]
    real_det = sim_det
    gap = calculate_perception_sim2real_gap(sim_det, real_det)
    print('sim2real keys:', list(gap.keys()))
    weather = calculate_weather_simulation_quality({'rain': sim_images}, {'rain': real_images}, weather_type='rain')
    print('weather keys:', list(weather.keys()))

# 9) Utilities
def block_utils():
    from admetrics.utils.matching import greedy_matching
    from admetrics.utils.nms import nms_bev
    from admetrics.utils.transforms import convert_box_format, transform_box
    from admetrics.utils.visualization import plot_boxes_bev
    preds = [{'box': np.array([0,0,0,1,1,1,0]), 'score':0.9}]
    gts = [{'box': np.array([0,0,0,1,1,1,0])}]
    matches = greedy_matching(preds, gts)
    print('matches:', matches)
    # nms_bev expects a list of detection dicts with 'box' keys
    kept = nms_bev([{'box': np.array([0,0,0,1,1,1,0]), 'score': 0.9}])
    print('nms kept:', kept)
    box = np.array([0,0,0,1,1,1,0])
    cf = convert_box_format(box, src_format='xyzwhlr', dst_format='xyzwhlr')
    print('convert box ok')


blocks = [
    ('detection', block_detection),
    ('tracking', block_tracking),
    ('prediction', block_prediction),
    ('localization', block_localization),
    ('occupancy', block_occupancy),
    ('vectormap', block_vectormap),
    ('planning', block_planning),
    ('simulation', block_simulation),
    ('utils', block_utils),
]

for name, fn in blocks:
    run_block(name, fn)

print('\nSummary:')
if errors:
    print(f'{len(errors)} blocks failed: {[e[0] for e in errors]}')
    raise SystemExit(1)
else:
    print('All blocks ran successfully')
    raise SystemExit(0)

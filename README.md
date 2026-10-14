We have a short sprint, so the goal should not be a production warehouse system. A strong outcome is a working prototype using two recorded camera videos, with local tracking IDs, basic cross-camera matching, positions on a 2D floor map, and visible movement trails.

1. Understand the system you are building

The eventual pipeline will look like this:

```
Camera streams
      ↓
Person detection
      ↓
Single-camera tracking
      ↓
Appearance embedding extraction
      ↓
Cross-camera Re-ID and global ID assignment
      ↓
Camera-to-floor coordinate projection
      ↓
Global position fusion
      ↓
2D map with IDs and trajectory history
```

Keep these three identity concepts separate:

Detection: “There is a person in this frame.”
Single-camera tracking: “This is track 7 in camera 1.”
Cross-camera Re-ID: “Camera 1 track 7 and camera 2 track 12 are probably the same person.”
Global ID: “Both tracks belong to warehouse person ID 3.”

A practical multi-camera system must handle occlusion, perspective changes, illumination changes and open-set identities, meaning people may enter without already existing in a fixed gallery. Multi-Camera Industrial Open-Set Person Re-Identification and Tracking discusses these industrial and open-set constraints directly.

Roadmap: today through Saturday
Tonight, Tuesday: preparation, 30–45 minutes
Objective

Prepare your environment and collect suitable test material.

Tasks
Create the project structure:

```
warehouse-reid/
├── videos/
│   ├── camera_01.mp4
│   └── camera_02.mp4
├── maps/
│   └── warehouse_map.png
├── calibration/
├── outputs/
├── src/
│   ├── detect_track.py
│   ├── reid.py
│   ├── calibration.py
│   ├── global_tracker.py
│   └── visualize_map.py
└── requirements.txt
```

Prepare two videos:

Prefer synchronized recordings.
Include at least two people.
Make sure one person appears in both cameras.
Begin with short clips, around one or two minutes.
Fixed cameras are much easier than moving cameras.

Create a simple warehouse map:

It can be a screenshot, floor plan or hand-drawn rectangle.
Mark walls, doors and camera positions.
Exact scale is not required for the first prototype.

Set up a Python environment:

```Python
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```PowerShell
.venv\Scripts\activate
```

Tonight’s deliverable
Project folder created
Two videos available
Basic 2D map image available
Python environment working

## Wednesday: detection and single-camera tracking

### Study, about 2 hours

Learn these concepts:

1. Bounding boxes and confidence scores
2. Intersection over Union, or IoU
3. Non-maximum suppression
4. Tracking-by-detection
5. Kalman filters
6. Hungarian assignment
7. Difference between ByteTrack and appearance-assisted trackers

The YOLO Multi-Object Tracking in Video documentation explains that tracking extends detection by maintaining object IDs across video frames and provides Python interfaces for processing videos and streams.

### Build, about 3–4 hours

Start with one camera only.

Required behavior:

* Detect only the person class.
* Assign a local track ID.
* Draw the bounding box.
* Draw the ID.
* Store the bottom-center point of each bounding box.
* Save recent points as a trajectory.

Use the bottom-center rather than the bounding-box center:

```Python
foot_x = (x1 + x2) / 2
foot_y = y2
```

That point is an approximation of the person’s contact position on the floor.

Store observations in a structure like:

```Python
observation = {
    "camera_id": 1,
    "local_track_id": 7,
    "timestamp": 12.4,
    "bbox": [x1, y1, x2, y2],
    "foot_point": [foot_x, foot_y],
    "confidence": 0.91
}
```

### Suggested experiment

Run ByteTrack first, then compare it with an appearance-aware tracker.

For this sprint, evaluate:

* Does the ID survive a short occlusion?
* Does the ID change when two people cross?
* Does a person receive a new ID after leaving and returning?
* How many obvious ID switches occur?

### Wednesday deliverable

A saved video for each camera showing:

* Person boxes
* Stable local IDs
* Short trajectory lines
* Basic FPS display

### Do not work on yet

* Cross-camera IDs
* Model training
* Face recognition
* Database or cloud deployment
* Live RTSP cameras

## Thursday: Re-ID fundamentals and cross-camera matching

Study, about 2–3 hours

Focus on the Re-ID process:

Person crop
    ↓
Re-ID neural network
    ↓
Feature embedding
    ↓
Distance comparison
    ↓
Same person or different person

A Re-ID embedding is a vector representing visual appearance. You compare two normalized embeddings using cosine similarity or Euclidean distance.

Cosine similarity:

similarity = embedding_a @ embedding_b

For normalized vectors:

Larger similarity means more visually similar.
A threshold decides whether a match is accepted.
The correct threshold must be tested on your own videos.

Learn these terms:

Query and gallery
Embedding
Positive and negative pair
Cosine similarity
Rank-1 accuracy
Mean Average Precision
Intra-class variation
Inter-class similarity
Domain shift
Closed-set versus open-set Re-ID
Build, about 3–4 hours

For each stable local track:

Collect several good person crops.
Reject very small, blurry or heavily occluded crops.
Extract an embedding for every accepted crop.
Average the normalized embeddings:
track_embedding = mean(normalized_crop_embeddings)
track_embedding = normalize(track_embedding)

Compare tracks between camera 1 and camera 2.
Create a temporary global ID.

Use more than appearance alone. Your first matching score can combine:

total cost =
    appearance cost
    + temporal cost
    + camera-transition penalty

Example logic:

if appearance_similarity > threshold:
    if transition_time_is_possible:
        assign_same_global_id()

Do not accept an appearance match that is physically impossible. For example, if two cameras do not overlap, the same person normally cannot appear in both at exactly the same time.

Create a camera transition model
camera_graph = {
    1: {
        2: {
            "min_seconds": 2.0,
            "max_seconds": 30.0
        }
    }
}

These values are only placeholders. Measure real walking times in the warehouse later.

Thursday deliverable

Generate a CSV similar to:

camera_id,local_track_id,global_id,start_time,end_time
1,7,3,10.2,18.5
2,12,3,21.1,30.4

Also create a simple comparison report:

Camera 1 track 7 → Camera 2 track 12
Similarity: 0.84
Time gap: 2.6 seconds
Decision: MATCH
Global ID: 3

A useful research reference is ReidTrack: Reid-Only Multi-Target Multi-Camera Tracking, which approaches multi-camera tracking by clustering appearance embeddings and addresses identity switches caused by imperfect person detections.

Friday: camera calibration and the 2D warehouse map
Study, about 1–2 hours

Learn:

Image coordinates
World or map coordinates
Perspective transformation
Planar homography
Calibration point correspondence
Reprojection error

For a flat warehouse floor, a homography maps a point from the camera image into a shared floor plane:

camera foot point (u, v)
          ↓ homography
warehouse position (x, y)

Important limitation: a homography models a plane. It works best when you map the person’s estimated foot point to the floor, not their head or bounding-box center.

Build, about 4 hours

For each camera:

Choose at least four visible floor landmarks.
Record their pixel coordinates in the camera image.
Mark the corresponding coordinates on the warehouse map.
Calculate the homography matrix.
Save it:
calibration/camera_01_homography.npy
calibration/camera_02_homography.npy

Transform every person foot point.
Draw the transformed position on the shared map.

Conceptual code:

map_point = perspective_transform(
    camera_foot_point,
    camera_homography
)

Validate calibration

For each calibration landmark:

Project its camera coordinate onto the map.
Compare it with its known map coordinate.
Draw both points.
Calculate pixel or metric error.
Recalibrate if errors are large or highly inconsistent.
Handle overlapping cameras

If both cameras see the same global ID simultaneously, you may receive two map positions. For the prototype:

Reject positions outside the valid floor polygon.
Prefer the camera with the larger, clearer person box.
Alternatively, average positions only when they are close.
Never average two distant positions without investigating the identity match.

Multi-Camera Person Tracking with Top-Down Localization demonstrates the same broad structure: camera detections, camera-specific homographies, a unified top-down view, duplicate handling and global trajectory visualization. Use it as an architectural example rather than blindly copying its identity logic.

Friday deliverable

A map video or display such as:

ID 3 ●────────────
          previous positions

ID 8 ●──────

Each visible person should have:

Global ID
Current 2D position
Colored trail
Last-seen time
Current source camera
Saturday: integration, evaluation and demonstration
Morning: integrate the pipeline

Connect the modules:

Read camera frames
→ detect people
→ update local trackers
→ extract/update Re-ID embeddings
→ resolve global IDs
→ project foot points
→ update global trajectories
→ render camera views and map

Use a central record:

global_tracks = {
    3: {
        "last_seen": 24.8,
        "camera_id": 2,
        "map_position": [420, 180],
        "trajectory": [],
        "embedding": None,
        "active_local_tracks": {}
    }
}

Afternoon: test failure cases

Create a small manual ground-truth sheet. Test at least:

One person walking between cameras
Two people crossing
Partial occlusion behind a shelf
Similar clothing
Person leaving and returning
Simultaneous appearance in overlapping cameras
False person detections
Person standing near the edge of a camera

Record:

Test case
Expected global ID
Predicted global ID
ID switches
Missed detections
Map-position problems
Notes

Saturday success criteria

Your prototype is successful if it can:

Process two prerecorded camera videos
Detect and track people within each video
Assign local camera IDs
Match at least some identities across cameras
Project positions onto one map
Draw trajectory histories
Export tracking events to CSV or JSON
Demonstrate failure cases honestly

Do not define success as perfect Re-ID accuracy. Four days is enough to understand and demonstrate the architecture, but not enough to solve every domain-specific calibration, association and deployment problem.

Recommended technical scope
Sprint prototype
Input:    Two prerecorded MP4 videos
Detector: Pretrained person detector
Tracker:  ByteTrack or BoT-SORT-style tracker
Re-ID:    Pretrained person embedding model
Map:      Static PNG floor plan
Mapping:  One homography per camera
Output:   Annotated video + 2D map + CSV

Later production version
Input: RTSP streams
Synchronization: camera timestamps
Inference: GPU or edge device
Tracking: one local tracker per stream
Identity: global association service
Storage: event and trajectory database
UI: browser-based live floor map
Monitoring: FPS, dropped frames and camera health
Privacy: retention, access and anonymization controls

What to learn after Saturday

Study in this order:

Better synchronization

Camera timestamps
Frame buffering
Asynchronous streams

Global data association

Hungarian assignment
Tracklet clustering
Camera topology
Temporal constraints

Stronger Re-ID

Fine-tuning on warehouse footage
Hard-negative mining
Clothing and viewpoint changes
Open-set threshold calibration

Better spatial localization

Camera intrinsic and extrinsic calibration
Ground-plane geometry
Multi-view position fusion
Reprojection-error analysis

Evaluation

ID switches
IDF1
HOTA
MOTA
Re-ID rank metrics
Position error in metres

Deployment

Stream reconnection
GPU batching
Logging and monitoring
Privacy and access controls
Recommended time allocation
20% reading and watching explanations
60% implementation
20% testing, notes and debugging

The most important decision is to keep the target small: two recorded cameras, one shared map and a few correctly maintained global identities. This gives us an end-to-end foundation that can later be replaced module by module with stronger models and production infrastructure.

# Weeding_robort

AI validation workspace for the Autonomous Precision Weed Extraction Rover.
The current implementation focuses on the first AI phase: collect selected-crop
images, annotate a small YOLO dataset, train a lightweight YOLO26 model, and
validate `~crop` working points against extraction zones before connecting rover
actuators.

## AI Validation Quick Start

1. Create a Python environment and install AI dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements-ai.txt
   ```

2. Capture onion or groundnut images:

   ```powershell
   python -m scripts.capture_dataset --crop onion --condition sunlight
   python -m scripts.capture_dataset --crop groundnut --condition shadow
   ```

3. Annotate the selected crop as YOLO `crop`, then place the export in
   `datasets/crop_non_crop`.

   - To extract the latest exported dataset zip into a working folder, run:

     ```powershell
     python -m scripts.use_latest_zip_dataset
     # or to overwrite an existing working folder
     python -m scripts.use_latest_zip_dataset --force
     ```

4. Audit labels and train the initial lightweight model:

   ```powershell
   python -m scripts.prepare_selected_crop_dataset
   python -m scripts.audit_yolo_dataset
   python -m scripts.train_yolo --task segment --epochs 100 --imgsz 768 --name selected_crop_recall
   ```

5. Validate detections and four extraction zones from a camera or field image:

   ```powershell
   python -m scripts.run_detection --model runs/ai_validation/selected_crop_recall/weights/best.pt
   python -m scripts.run_detection --model runs/ai_validation/selected_crop_recall/weights/best.pt --source sample.jpg --conf 0.25 --guard-px 32
   ```

See [docs/ai_validation.md](docs/ai_validation.md) for annotation rules,
sunlight/shadow checks, segmentation training, and the current actuator handoff
boundary.

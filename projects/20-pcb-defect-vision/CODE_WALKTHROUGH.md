# Code walkthrough

Read in this order. Paths are relative to this project folder.

1. **`src/prepare_patches.py`**: clones DeepPCB at a pinned commit into
   `downloads/`. It reads `trainval.txt` and `test.txt`, and for every
   annotated box cuts a 64×64 crop centred on the defect from the
   *tested* image and the same window from the *template*. It adds 3
   random defect-free crops per image that keep 4 px clear of any box.
   The last two trainval image groups become validation. It writes
   `data/patches/{train,val,test}.npz` and `data/patch_metadata.json`
   (counts, source, licence, citation).

2. **`src/vision_lib.py`**: shared code.
   - `to_tensor()`: uint8 greyscale to normalised 3-channel tensors at the
     model's input size (64 px for the small CNN, 96 px for ResNet-18).
   - `augment()`: flips and 90° rotations (PCB defects have no
     orientation).
   - `SmallCNN`: four conv/BN/ReLU/pool blocks and global pooling.
   - `make_resnet()`: torchvision ResNet-18 with a new 7-class head.
   - `gradcam()`: forward/backward hooks on `layer3`, channel weights =
     mean gradient, ReLU, upsample, normalise to 0–1.

3. **`src/train.py`**: AdamW with a one-cycle schedule, inverse-frequency
   class weights, augmentation per batch, 6 epochs. It keeps the epoch
   with the best validation macro-F1 and saves `models/<name>.pt` and
   `reports/training_log.json`.

4. **`src/evaluate.py`**: everything scored once on test.
   - `classifier_report()`: accuracy, macro-F1, per-class P/R/F1 and the
     confusion matrix, plus seen vs unseen image groups.
   - `binary_at()`: one helper for every method. It picks the defect/OK
     threshold with the best F1 on validation, then reports test AUROC,
     recall, missed defects and false alarms.
   - `template_baseline()`: grid over smoothing and grey-level threshold,
     chosen by AUROC on train, with false-alarm rates per test group.
   - `patchcore()`: frozen ResNet-18 layer2 + layer3 features on a 6×6
     grid. The memory bank holds 10,000 random vectors from OK training
     patches. Score = max over the grid of the nearest-neighbour distance
     (`torch.cdist`).
   - It saves the confusion-matrix and Grad-CAM charts and
     `reports/metrics.json`.

5. **`src/board_scan.py`**: slides a 64 px window (stride 32, edges
   included) over a full board, classifies every window and averages
   P(defect) into a heat map. `available_model()` prefers ResNet-18 and
   falls back to the committed small CNN.

6. **`app.py`**: the Streamlit demo. Pick a DeepPCB board or upload one;
   it shows the heat map and the windows above a chosen threshold.

7. **`tests/test_vision.py`**: preprocessing, augmentation, shapes,
   Grad-CAM range, scan coverage, the template rule, `metrics.json`
   consistency, the model loader, and an app render.

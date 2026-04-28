# ROI-VFC training/testing pipeline (PyTorch)

Tài liệu này mô tả nhanh cách dùng các module mới:

- `src/datasets/train_dataset.py`
- `src/datasets/test_dataset.py`
- `src/trainers/train_roi_vfc.py`
- `src/trainers/test_roi_vfc.py`

## 1) Định dạng dữ liệu đầu vào

Manifest có thể là `list[dict]` hoặc file JSON với dạng:

```json
{
  "samples": [
    {
      "video_id": "vid_001",
      "frames": ["/abs/path/f0001.jpg", "/abs/path/f0002.jpg"],
      "labels": {
        "segmentation_mask_path": "/abs/path/mask_0002.png",
        "detection": "... optional task label ..."
      },
      "target_features_path": "/abs/path/target_feature.pt"
    }
  ]
}
```

Hoặc mỗi sample dùng video trực tiếp:

```json
{
  "video": "/abs/path/video.mp4",
  "num_frames": 120,
  "labels": {"segmentation_mask_path": "/abs/path/mask.png"}
}
```

## 2) Output của dataset

Mỗi sample trả về:

- `frames`: tensor `[T, C, H, W]`
- `labels`: `list[dict]` label theo từng frame (đã align theo chỉ số thời gian)
- `target_features`: `list[Tensor|None]` theo từng frame
- `frame_indices`: chỉ số frame gốc được lấy trong video
- `length`: số frame hợp lệ trong sample
- `video_id`: string

Collate function trả về batch:

- `frames`: `[B, T, C, H, W]`
- `frame_mask`: `[B, T]` (`True` cho timestep hợp lệ, `False` cho padding)
- `lengths`: `list[int]`
- `frame_indices`: `list[list[int]]`
- `labels`: `list[list[dict|None]]` (đã pad theo `T`)
- `target_features`: `list[list[Tensor|None]]` (đã pad theo `T`)
- `video_ids`: `list[str]`

> Ghi chú: collate sẽ pad clip ngắn bằng cách lặp frame cuối để giữ shape, và dùng `frame_mask` để trainer bỏ qua loss ở timestep padding.

## 3) Train/Test API

Trong `src/trainers/train_roi_vfc.py`:

- `train_roi_vfc_epoch(...)`
- `test_roi_vfc_epoch(...)`
- `LossWeights`

Pipeline tối ưu gồm:

1. Loop theo thời gian `t=0..T-1` cho từng clip, reset buffer đầu clip.
2. `frame_t -> FeatureExtraction -> ROI_VFC -> FeatureSpaceTransfer`.
3. Dùng `labels[b][t]` và `target_features[b][t]` để align loss downstream theo frame.
4. Bỏ qua timestep padding bằng `frame_mask[b][t]`.
5. Cộng loss theo thời gian:
  - ROI-VFC rate loss (`bpp`)
  - ROI-VFC feature reconstruction loss
  - Feature transfer domain loss
  - Feature transfer image reconstruction loss
  - Downstream supervised loss (optional)
  - Downstream consistency MSE (feature gốc vs feature mới)

Trong `src/trainers/test_roi_vfc.py`:

- `evaluate_roi_vfc_downstream(...)`

Hàm này hỗ trợ evaluate detection/segmentation qua criterion hoặc metric custom.

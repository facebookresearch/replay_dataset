import logging
import os
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Generic, Optional, Tuple, Type, TypeVar

import cv2
import numpy as np
import sqlalchemy as sa
import torch

from pytorch3d.implicitron.dataset.frame_data import FrameData, GenericFrameDataBuilder

from pytorch3d.implicitron.dataset.orm_types import (
    SqlFrameAnnotation,
)
from pytorch3d.implicitron.dataset.sql_dataset import SqlIndexDataset
from pytorch3d.implicitron.dataset.utils import (
    GenericWorkaround,
    get_bbox_from_mask,
    load_image,
    load_pointcloud,
    safe_as_tensor,
    transpose_normalize_image,
)
from pytorch3d.implicitron.tools.config import registry

from .replay_orm_types import (
    ReplayFrameAnnotation,
    ReplaySequenceAnnotation,
    ReplaySensorAnnotation,
)

logger = logging.getLogger(__name__)


# TODO: move to utils
K = TypeVar("K")
T = TypeVar("T")


@dataclass
class LruCacheWithCleanup(Generic[K, T]):
    """ Requires Python 3.6+ since it assumes insertion-ordered dict.
    """
    create_fn: Callable[[K], T]
    cleanup_fn: Callable[[T], None] = field(default=lambda key: None)
    max_size: int | None = None
    _cache: dict[K, T] = field(init=False, default_factory=lambda: {})

    def __getitem__(self, key: K) -> T:
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value  # update the order
            return value

        # inserting a new element
        if self.max_size and len(self._cache) >= self.max_size:
            # need to clean up the oldest element
            oldest_key = next(iter(self._cache))
            oldest_value = self._cache.pop(oldest_key)
            # TODO: uncomment when moved to utils
            #logger.debug(f"Releasing an object for {oldest_key}")
            self.cleanup_fn(oldest_value)

        assert self.max_size is None or len(self._cache) < self.max_size
        #logger.debug(f"Creating an object for {key}")
        value = self.create_fn(key)
        self._cache[key] = value
        return value

    def cleanup_all(self) -> None:
        for value in self._cache.values():
            self.cleanup_fn(value)

        self._cache = {}


@dataclass
class ReplayFrameData(FrameData):
    sensor_name: str = ""


# TODO: this can be a generic VideoSamplingFrameDataBuilder[FrameDataType] class
@registry.register
class ReplayFrameDataBuilder(
    GenericWorkaround, GenericFrameDataBuilder[ReplayFrameData]
):
    """
    A concrete class to build an extended FrameData object
    """
    load_depths: bool = False  # override
    frame_data_type: ClassVar[Type[FrameData]] = ReplayFrameData
    load_frames_from_videos: bool = True
    video_capture_cache_size: int = 16

    def __post_init__(self) -> None:
        super().__post_init__()

        self._video_capture_cache = LruCacheWithCleanup[str, cv2.VideoCapture](
            create_fn=lambda path: cv2.VideoCapture(path),
            cleanup_fn=lambda capture: capture.release(),
            max_size=self.video_capture_cache_size,
        )

    def build(
        self,
        frame_annotation: ReplayFrameAnnotation,
        sequence_annotation: ReplaySequenceAnnotation,
        *,
        sensor_annotation: ReplaySensorAnnotation,
        load_blobs: bool = True,
    ) -> ReplayFrameData:
        # TODO: redo frame loading from video
        point_cloud = sequence_annotation.point_cloud

        frame_data = self.frame_data_type(
            frame_number=safe_as_tensor(frame_annotation.frame_number, torch.long),
            frame_timestamp=safe_as_tensor(
                frame_annotation.frame_timestamp, torch.float
            ),
            sequence_name=frame_annotation.sequence_name,
            sequence_category=sequence_annotation.category,
            camera_quality_score=safe_as_tensor(
                sequence_annotation.viewpoint_quality_score, torch.float
            ),
            point_cloud_quality_score=safe_as_tensor(
                point_cloud.quality_score, torch.float
            )
            if point_cloud is not None
            else None,
            sensor_name=frame_annotation.sensor_name,
        )

        mask_annotation = frame_annotation.mask
        if mask_annotation is not None:
            fg_mask_np: Optional[np.ndarray] = None
            if load_blobs and self.load_masks and mask_annotation.mass:
                if self.load_frames_from_videos:
                    fg_mask_np = self._frame_from_video(
                        sensor_annotation.mask_file_path,
                        frame_annotation.frame_timestamp
                        - sensor_annotation.timestamp_delta,
                    )
                    fg_mask_np = fg_mask_np[:1]  # OpenCV converts grayscale to RGB

                if fg_mask_np is None:
                    fg_mask_np, mask_path = self._load_fg_probability(frame_annotation)
                    frame_data.mask_path = mask_path

                frame_data.fg_probability = safe_as_tensor(fg_mask_np, torch.float)

            bbox_xywh = mask_annotation.bounding_box_xywh
            if bbox_xywh is None and fg_mask_np is not None:
                bbox_xywh = get_bbox_from_mask(fg_mask_np, self.box_crop_mask_thr)

            frame_data.bbox_xywh = safe_as_tensor(bbox_xywh, torch.float)

        if frame_annotation.image is not None:
            image_size_hw = safe_as_tensor(frame_annotation.image.size, torch.long)
            frame_data.image_size_hw = image_size_hw  # original image size
            # image size after crop/resize
            frame_data.effective_image_size_hw = image_size_hw
            if frame_annotation.image.path is not None and self.dataset_root is not None:
                frame_data.image_path = os.path.join(
                    self.dataset_root, frame_annotation.image.path
                )

            if load_blobs and self.load_images:
                assert frame_data.image_path is not None
                image_np = self._frame_from_video(
                    sensor_annotation.file_path,
                    frame_annotation.frame_timestamp - sensor_annotation.timestamp_delta,
                ) if self.load_frames_from_videos else None
                if image_np is None:  # fall back to loading from images
                    image_np = load_image(self._local_path(frame_data.image_path))
                frame_data.image_rgb = self._postprocess_image(
                    image_np, frame_annotation.image.size, frame_data.fg_probability
                )

        if (
            load_blobs
            and self.load_depths
            and frame_annotation.depth is not None
            and frame_annotation.depth.path is not None
        ):
            (
                frame_data.depth_map,
                frame_data.depth_path,
                frame_data.depth_mask,
            ) = self._load_mask_depth(frame_annotation, frame_data.fg_probability)

        if load_blobs and self.load_point_clouds and point_cloud is not None:
            pcl_path = self._fix_point_cloud_path(point_cloud.path)
            frame_data.sequence_point_cloud = load_pointcloud(
                self._local_path(pcl_path), max_points=self.max_points
            )
            frame_data.sequence_point_cloud_path = pcl_path

        if frame_annotation.viewpoint is not None:
            frame_data.camera = self._get_pytorch3d_camera(frame_annotation)

        if self.box_crop:
            frame_data.crop_by_metadata_bbox_(self.box_crop_context)

        if self.image_height is not None and self.image_width is not None:
            new_size = (self.image_height, self.image_width)
            frame_data.resize_frame_(
                new_size_hw=torch.tensor(new_size, dtype=torch.long),  # pyre-ignore
            )

        return frame_data

    def _frame_from_video(
        self, video_path: str | None, timestamp_sec: float
    ) -> np.ndarray | None:
        assert self.dataset_root is not None
        if not video_path or not self._exists_in_dataset_root(video_path):
            logger.debug(f"Video path not found at {video_path}; using image path.")
            return None

        if timestamp_sec < -1e-2:
            logger.debug(f"Trying to get a frame at {timestamp_sec} s from {video_path}")
            return None

        path = self._local_path(os.path.join(self.dataset_root, video_path))
        capture = self._video_capture_cache[path]
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
        ret, image = capture.read()
        if not ret:
            logger.warning(f"Failed to get frame from {video_path} at {timestamp_sec}.")
            return None

        return transpose_normalize_image(image)


class ReplayDataset(SqlIndexDataset):
    frame_annotations_type: ClassVar[
        Type[SqlFrameAnnotation]
    ] = ReplayFrameAnnotation

    frame_data_builder_class_type: str = "ReplayFrameDataBuilder"

    # override
    def _get_item(
        self, frame_idx: int | Tuple[str, int | torch.LongTensor], load_blobs: bool = True
    ) -> ReplayFrameData:
        match frame_idx:
            case int():
                if frame_idx >= len(self._index):
                    raise IndexError(f"index {frame_idx} out of range {len(self._index)}")

                seq, frame = self._index.index[frame_idx]
            case seq, frame, *rest:
                if isinstance(frame, torch.LongTensor):
                    frame = frame.item()

                if (seq, frame) not in self._index.index:
                    raise IndexError(
                        f"Sequence-frame index {frame_idx} not found; is it filtered out?"
                    )

                if rest and rest[0] != self._index.loc[(seq, frame), "_image_path"]:
                    raise IndexError(f"Non-matching image path in {frame_idx}.")

        FrameAnnotation = self.frame_annotations_type
        stmt = sa.select(
            FrameAnnotation, ReplaySequenceAnnotation, ReplaySensorAnnotation
        ).join(
            ReplaySequenceAnnotation,
            FrameAnnotation.sequence_name == ReplaySequenceAnnotation.sequence_name,
        ).join(
            ReplaySensorAnnotation, sa.and_(
                FrameAnnotation.sequence_name == ReplaySensorAnnotation.sequence_name,
                FrameAnnotation.sensor_name == ReplaySensorAnnotation.sensor_name,
            )
        ).where(
            FrameAnnotation.sequence_name == seq,
            FrameAnnotation.frame_number == int(frame),  # cast from np.int64
        )

        with sa.orm.Session(self._sql_engine) as session:
            entry, seq_metadata, sensor_metadata = session.execute(stmt).one()

        assert entry.image.path == self._index.loc[(seq, frame), "_image_path"]

        frame_data = self.frame_data_builder.build(
            entry, seq_metadata, sensor_annotation=sensor_metadata, load_blobs=load_blobs
        )

        # The rest of the fields are optional
        frame_data.frame_type = self._get_frame_type(entry)
        return frame_data

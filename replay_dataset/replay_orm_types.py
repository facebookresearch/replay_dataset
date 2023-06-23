import numpy as np

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from pytorch3d.implicitron.dataset.orm_types import (
    ArrayTypeFactory,
    Base,
    SqlFrameAnnotation,
    SqlSequenceAnnotation,
    TupleTypeFactory,
)


class ReplayFrameAnnotation(SqlFrameAnnotation):
    sensor_name: Mapped[str] = mapped_column(nullable=True, index=True)


class ReplaySequenceAnnotation(SqlSequenceAnnotation):
    scenario: Mapped[str] = mapped_column(nullable=True)
    space_id: Mapped[int] = mapped_column(nullable=True)  # as in scene.json


class ReplaySensorAnnotation(Base):
    __tablename__ = "sensor_annots"

    # This composite key can be used to join to FrameAnnotation
    sensor_name: Mapped[str] = mapped_column(primary_key=True)
    sequence_name: Mapped[str] = mapped_column(primary_key=True)

    sensor_type: Mapped[str]  # "video" | "audio"

    file_path: Mapped[str] = mapped_column(nullable=True)
    # To determine the timestamp in file_path reference frame:
    #   timestamp_local = frame_timestamp - timestamp_delta
    timestamp_delta: Mapped[float] = mapped_column(nullable=True)

    # the below only make sense for the sensor_type == video:
    mask_file_path: Mapped[str] = mapped_column(nullable=True)
    depth_file_path: Mapped[str] = mapped_column(nullable=True)
    depth_mask_file_path: Mapped[str] = mapped_column(nullable=True)


class ReplayActorAnnotation(Base):
    __tablename__ = "actor_annots"

    actor_id: Mapped[int] = mapped_column(primary_key=True)  # taken from vendor’s JSON
    gender: Mapped[str] = mapped_column(nullable=True)
    age: Mapped[int] = mapped_column(nullable=True)
    ethnicity: Mapped[str] = mapped_column(nullable=True)
    skin_tone: Mapped[str] = mapped_column(nullable=True)


class ReplayActorFrameAnnotation(Base):
    __tablename__ = "actor_frame_annots"

    actor_id: Mapped[int] = mapped_column(primary_key=True)
    sequence_name: Mapped[str] = mapped_column(primary_key=True)
    frame_number: Mapped[int] = mapped_column(primary_key=True)

    # TODO: fill?? Or maybe sensor_name is enough
    #nearrange_microphone_sensor_name: Mapped[str] = mapped_column(nullable=True)

    bounding_box_xywh: Mapped[tuple[float, float, float, float]] = mapped_column(
        TupleTypeFactory(float, shape=(4,)), nullable=True
    )

    # human pose (e.g. keypoints) in image coordinates, N×2 array
    pose_2d: Mapped[np.ndarray] = mapped_column(ArrayTypeFactory(), nullable=True)
    # 3D human pose in world coordinates, e.g. SMLP vertices, N×3 array
    pose_3d: Mapped[np.ndarray] = mapped_column(ArrayTypeFactory(), nullable=True)

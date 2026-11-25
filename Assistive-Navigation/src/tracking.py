"""Object tracking and track-level events.

Two layers live here:

1. **Identity** - Ultralytics' ByteTrack assigns ``track_id`` inside the model
   call. When tracking is unavailable (no ``lap``, or ``--no-track``) the
   :class:`TrackManager` assigns IDs itself with a small class-aware IoU
   matcher, so the rest of the program always has stable IDs.

2. **Lifecycle** - :class:`TrackManager` turns per-frame detections into
   events: ``appeared``, ``direction_change`` and ``disappeared``. The voice
   module only listens to these events, which is what stops it from repeating
   the same sentence every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from direction import Box, Direction


class DetectionLike(Protocol):
    """Minimal shape the tracker needs from a detection."""

    class_id: int
    name: str
    confidence: float
    box: Box
    track_id: int | None
    direction: Direction | None


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackEvent:
    """Something worth telling the user about."""

    kind: str  # "appeared" | "direction_change" | "disappeared"
    track_id: int
    class_id: int
    name: str
    confidence: float
    direction: Direction | None = None
    previous_direction: Direction | None = None
    box: Box | None = None
    frame_index: int = 0
    # Filled by the main loop when approximate distance reporting is enabled.
    distance_m: float | None = None


@dataclass
class TrackInfo:
    """State kept for one tracked object."""

    track_id: int
    class_id: int
    name: str
    confidence: float
    box: Box
    direction: Direction | None
    first_frame: int
    last_frame: int
    hits: int = 1
    missing: int = 0
    announced_direction: Direction | None = None
    announced_frame: int = -1
    spoken_count: int = 0

    @property
    def is_confirmed(self) -> bool:
        return self.missing == 0

    def age_frames(self) -> int:
        return self.last_frame - self.first_frame + 1


class TrackManager:
    """Maintain track identity, lifetime and direction-change events.

    Parameters
    ----------
    max_missing:
        Frames a track may be absent before it is dropped and ``disappeared``
        is emitted. 15 frames is about half a second at 30 fps.
    iou_threshold:
        Overlap needed by the fallback matcher to call two boxes the same
        object.
    """

    def __init__(self, max_missing: int = 15, iou_threshold: float = 0.3) -> None:
        if max_missing < 1:
            raise ValueError("max_missing must be >= 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        self.max_missing = int(max_missing)
        self.iou_threshold = float(iou_threshold)
        self._tracks: dict[int, TrackInfo] = {}
        self._next_id = 1
        self._frame = 0

    # -- read-only views ----------------------------------------------------
    @property
    def tracks(self) -> dict[int, TrackInfo]:
        return self._tracks

    def active(self) -> list[TrackInfo]:
        """Tracks seen in the most recent frame."""
        return [t for t in self._tracks.values() if t.missing == 0]

    def get(self, track_id: int) -> TrackInfo | None:
        return self._tracks.get(track_id)

    # -- identity -----------------------------------------------------------
    def _fresh_id(self) -> int:
        """Next free fallback ID (never collides with an existing one)."""
        while self._next_id in self._tracks:
            self._next_id += 1
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def _assign_fallback_ids(self, detections: Sequence[DetectionLike]) -> None:
        """Greedy, class-aware IoU matching for detections without an ID."""
        unassigned = [d for d in detections if d.track_id is None]
        if not unassigned:
            return
        candidates: list[tuple[float, int, int]] = []
        for di, det in enumerate(unassigned):
            for track in self._tracks.values():
                if track.missing > self.max_missing or track.class_id != det.class_id:
                    continue
                score = iou(det.box, track.box)
                if score >= self.iou_threshold:
                    candidates.append((score, di, track.track_id))
        candidates.sort(reverse=True)
        used_dets: set[int] = set()
        used_tracks: set[int] = set()
        for _score, di, track_id in candidates:
            if di in used_dets or track_id in used_tracks:
                continue
            unassigned[di].track_id = track_id
            used_dets.add(di)
            used_tracks.add(track_id)
        for di, det in enumerate(unassigned):
            if det.track_id is None:
                det.track_id = self._fresh_id()

    # -- per-frame update ---------------------------------------------------
    def update(
        self, detections: Sequence[DetectionLike], frame_index: int | None = None
    ) -> list[TrackEvent]:
        """Match detections to tracks and return appeared/disappeared events."""
        self._frame = self._frame + 1 if frame_index is None else int(frame_index)
        events: list[TrackEvent] = []

        self._assign_fallback_ids(detections)
        seen: set[int] = set()

        for det in detections:
            track_id = det.track_id
            if track_id is None:  # defensive: matcher should always set one
                track_id = det.track_id = self._fresh_id()
            if track_id in seen:
                continue
            seen.add(track_id)

            track = self._tracks.get(track_id)
            if track is None:
                track = TrackInfo(
                    track_id=track_id,
                    class_id=det.class_id,
                    name=det.name,
                    confidence=det.confidence,
                    box=det.box,
                    direction=det.direction,
                    first_frame=self._frame,
                    last_frame=self._frame,
                )
                self._tracks[track_id] = track
                events.append(
                    TrackEvent(
                        kind="appeared",
                        track_id=track_id,
                        class_id=det.class_id,
                        name=det.name,
                        confidence=det.confidence,
                        direction=det.direction,
                        box=det.box,
                        frame_index=self._frame,
                    )
                )
            else:
                track.class_id = det.class_id
                track.name = det.name
                track.confidence = det.confidence
                track.box = det.box
                track.last_frame = self._frame
                track.hits += 1
                track.missing = 0
                if det.direction is not None:
                    track.direction = det.direction

        # Age out tracks that were not seen this frame.
        for track_id in list(self._tracks):
            if track_id in seen:
                continue
            track = self._tracks[track_id]
            track.missing += 1
            if track.missing > self.max_missing:
                events.append(
                    TrackEvent(
                        kind="disappeared",
                        track_id=track_id,
                        class_id=track.class_id,
                        name=track.name,
                        confidence=track.confidence,
                        direction=track.direction,
                        box=track.box,
                        frame_index=self._frame,
                    )
                )
                del self._tracks[track_id]

        return events

    def note_directions(
        self, detections: Iterable[DetectionLike]
    ) -> list[TrackEvent]:
        """Emit ``direction_change`` when an object crosses into a new zone.

        Call this after the smoothed direction has been written onto each
        detection, so a one-frame boundary wobble cannot trigger speech.
        """
        events: list[TrackEvent] = []
        for det in detections:
            # Prefer the smoothed zone so a one-frame boundary wobble cannot
            # produce a "moved to..." sentence.
            direction = getattr(det, "direction_stable", None) or det.direction
            if det.track_id is None or direction is None:
                continue
            track = self._tracks.get(det.track_id)
            if track is None:
                continue
            if track.announced_direction is None:
                track.announced_direction = direction
                continue
            if direction == track.announced_direction:
                continue
            previous = track.announced_direction
            track.announced_direction = direction
            events.append(
                TrackEvent(
                    kind="direction_change",
                    track_id=det.track_id,
                    class_id=det.class_id,
                    name=det.name,
                    confidence=det.confidence,
                    direction=direction,
                    previous_direction=previous,
                    box=det.box,
                    frame_index=self._frame,
                )
            )
        return events

    def mark_spoken(self, track_id: int, direction: Direction | None = None) -> None:
        """Record that this track was just announced."""
        track = self._tracks.get(track_id)
        if track is None:
            return
        track.spoken_count += 1
        track.announced_frame = self._frame
        if direction is not None:
            track.announced_direction = direction

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame = 0

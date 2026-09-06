import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

_SOURCE_FRAME_PATTERN = re.compile(
    r"(?:^|[_-])(?:f|frame)[_-]?(\d+)$",
    re.IGNORECASE,
)


def source_frame_number(path: str | Path) -> int | None:
    """Return a source frame number from common video-frame filenames."""
    stem = Path(path).stem
    if stem.isdigit():
        return int(stem)
    match = _SOURCE_FRAME_PATTERN.search(stem)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class FrameInfo:
    index: int
    path: Path
    name: str
    source_index: int | None = None


@dataclass
class FrameSequence:
    folder: Path
    frames: list[FrameInfo]
    _frames_by_name: dict[str, FrameInfo] = field(init=False, repr=False)

    def __post_init__(self):
        self._frames_by_name = {frame.name: frame for frame in self.frames}
        if len(self._frames_by_name) != len(self.frames):
            raise ValueError("Frame names must be unique within a sequence.")

    @classmethod
    def from_folder(cls, folder: str | Path) -> "FrameSequence":
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Directory not found: {folder}")

        paths = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        source_indices = [source_frame_number(path) for path in paths]
        if all(index is not None for index in source_indices):
            paths = [
                path
                for _, path in sorted(
                    zip(source_indices, paths),
                    key=lambda item: (item[0], item[1].name),
                )
            ]
            source_indices = [source_frame_number(path) for path in paths]
        else:
            paths.sort(key=lambda path: path.name)
            source_indices = list(range(len(paths)))
        if not paths:
            raise ValueError(f"No supported image frames found in: {folder}")

        return cls(
            folder=folder,
            frames=[
                FrameInfo(
                    index=i,
                    path=p,
                    name=p.name,
                    source_index=source_indices[i],
                )
                for i, p in enumerate(paths)
            ]
        )

    @classmethod
    def from_paths(
        cls,
        folder: str | Path,
        paths,
        source_indices=None,
    ) -> "FrameSequence":
        paths = [Path(path) for path in paths]
        if not paths:
            raise ValueError("A frame sequence requires at least one frame.")
        if source_indices is None:
            source_indices = list(range(len(paths)))
        else:
            source_indices = list(source_indices)
        if len(source_indices) != len(paths):
            raise ValueError("Frame paths and source indices must have equal lengths.")

        return cls(
            folder=Path(folder),
            frames=[
                FrameInfo(
                    index=index,
                    path=path,
                    name=path.name,
                    source_index=source_index,
                )
                for index, (path, source_index) in enumerate(
                    zip(paths, source_indices)
                )
            ],
        )

    def index_for_name(self, name: str) -> int | None:
        frame = self._frames_by_name.get(name)
        return frame.index if frame else None

    def name_for_index(self, index: int) -> str | None:
        if 0 <= index < len(self.frames):
            return self.frames[index].name
        return None

    def frame_for_name(self, name: str) -> FrameInfo | None:
        return self._frames_by_name.get(name)

    def end_index_for_max_gap(self, start_index: int, max_gap: int) -> int:
        """Find the last contiguous frame whose source gap stays in range."""
        if not 0 <= start_index < len(self.frames):
            raise IndexError("Start index is outside the frame sequence.")
        if max_gap < 1:
            raise ValueError("Maximum frame gap must be at least 1.")

        end_index = start_index
        for index in range(start_index, len(self.frames) - 1):
            current = self.frames[index].source_index
            following = self.frames[index + 1].source_index
            if current is None or following is None or following - current > max_gap:
                break
            end_index = index + 1
        return end_index

    def source_gap_after(self, index: int) -> int | None:
        if not 0 <= index < len(self.frames) - 1:
            return None
        current = self.frames[index].source_index
        following = self.frames[index + 1].source_index
        if current is None or following is None:
            return None
        return following - current

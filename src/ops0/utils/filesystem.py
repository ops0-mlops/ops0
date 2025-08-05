"""
ops0 Filesystem Utilities

Safe file operations, atomic writes, and filesystem helpers.
"""

import os
import shutil
import tempfile
import fcntl
import time
from pathlib import Path
from typing import Union, List, Optional, Callable, Iterator, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import logging
import fnmatch
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """File information"""
    path: Path
    size: int
    created: datetime
    modified: datetime
    is_file: bool
    is_dir: bool
    is_symlink: bool
    permissions: str

    @classmethod
    def from_path(cls, path: Path) -> 'FileInfo':
        """Create FileInfo from path"""
        stat = path.stat()

        return cls(
            path=path,
            size=stat.st_size,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            is_file=path.is_file(),
            is_dir=path.is_dir(),
            is_symlink=path.is_symlink(),
            permissions=oct(stat.st_mode)[-3:]
        )


def ensure_directory(
        path: Union[str, Path],
        mode: int = 0o755,
        parents: bool = True
) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        path: Directory path
        mode: Directory permissions
        parents: Create parent directories

    Returns:
        Path object

    Examples:
        >>> ensure_directory("/tmp/myapp/data")
        Path('/tmp/myapp/data')
    """
    path = Path(path)

    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")
        return path

    try:
        path.mkdir(mode=mode, parents=parents, exist_ok=True)
        logger.debug(f"Created directory: {path}")
        return path
    except Exception as e:
        raise IOError(f"Failed to create directory {path}: {e}")


def safe_file_write(
        path: Union[str, Path],
        content: Union[str, bytes],
        mode: str = 'w',
        encoding: str = 'utf-8',
        backup: bool = True
) -> Path:
    """
    Safely write file with optional backup.

    Args:
        path: File path
        content: Content to write
        mode: Write mode ('w' or 'wb')
        encoding: Text encoding
        backup: Create backup of existing file

    Returns:
        Path to written file

    Examples:
        >>> safe_file_write("/tmp/config.json", '{"key": "value"}')
        Path('/tmp/config.json')
    """
    path = Path(path)

    # Ensure parent directory exists
    ensure_directory(path.parent)

    # Backup existing file
    if backup and path.exists():
        backup_path = path.with_suffix(f"{path.suffix}.backup")
        shutil.copy2(path, backup_path)
        logger.debug(f"Created backup: {backup_path}")

    # Write file
    try:
        if 'b' in mode:
            path.write_bytes(content)
        else:
            path.write_text(content, encoding=encoding)

        logger.debug(f"Wrote file: {path}")
        return path

    except Exception as e:
        # Restore backup on failure
        if backup and backup_path.exists():
            shutil.move(backup_path, path)
            logger.warning(f"Restored backup after write failure: {path}")
        raise IOError(f"Failed to write file {path}: {e}")


@contextmanager
def atomic_write(
        path: Union[str, Path],
        mode: str = 'w',
        encoding: str = 'utf-8',
        **kwargs
):
    """
    Context manager for atomic file writes.

    Writes to a temporary file and moves it to final location on success.

    Args:
        path: Target file path
        mode: Write mode
        encoding: Text encoding
        **kwargs: Additional arguments for open()

    Examples:
        >>> with atomic_write("/tmp/data.json") as f:
        ...     json.dump(data, f)
    """
    path = Path(path)

    # Ensure parent directory exists
    ensure_directory(path.parent)

    # Create temporary file in same directory (for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )

    try:
        # Open temporary file
        if 'b' in mode:
            kwargs.pop('encoding', None)
            with os.fdopen(temp_fd, mode, **kwargs) as f:
                yield f
        else:
            kwargs['encoding'] = encoding
            with os.fdopen(temp_fd, mode, **kwargs) as f:
                yield f

        # Atomic rename
        os.replace(temp_path, path)
        logger.debug(f"Atomically wrote: {path}")

    except Exception:
        # Clean up temporary file on error
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def cleanup_old_files(
        directory: Union[str, Path],
        pattern: str = "*",
        days: int = 30,
        keep_count: Optional[int] = None,
        dry_run: bool = False
) -> List[Path]:
    """
    Clean up old files from directory.

    Args:
        directory: Directory to clean
        pattern: Filename pattern (glob)
        days: Remove files older than this many days
        keep_count: Keep at least this many recent files
        dry_run: Don't actually remove files

    Returns:
        List of removed file paths

    Examples:
        >>> cleanup_old_files("/tmp/logs", "*.log", days=7)
        [Path('/tmp/logs/old1.log'), Path('/tmp/logs/old2.log')]
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    # Find matching files
    files = []
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            stat = file_path.stat()
            files.append((file_path, stat.st_mtime))

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x[1], reverse=True)

    # Determine files to remove
    cutoff_time = time.time() - (days * 86400)
    removed = []

    for i, (file_path, mtime) in enumerate(files):
        # Keep minimum count
        if keep_count and i < keep_count:
            continue

        # Check age
        if mtime < cutoff_time:
            if not dry_run:
                try:
                    file_path.unlink()
                    logger.info(f"Removed old file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {e}")
                    continue

            removed.append(file_path)

    return removed


def find_files(
        directory: Union[str, Path],
        pattern: Optional[str] = None,
        recursive: bool = True,
        file_type: Optional[str] = None,  # 'file', 'dir', 'symlink'
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        modified_after: Optional[datetime] = None,
        modified_before: Optional[datetime] = None
) -> Iterator[Path]:
    """
    Find files matching criteria.

    Args:
        directory: Directory to search
        pattern: Filename pattern (glob or regex)
        recursive: Search subdirectories
        file_type: Filter by file type
        min_size: Minimum file size in bytes
        max_size: Maximum file size in bytes
        modified_after: Modified after this time
        modified_before: Modified before this time

    Yields:
        Matching file paths

    Examples:
        >>> list(find_files("/tmp", "*.log", min_size=1024))
        [Path('/tmp/app.log'), Path('/tmp/error.log')]
    """
    directory = Path(directory)

    # Determine glob method
    if recursive:
        glob_method = directory.rglob
    else:
        glob_method = directory.glob

    # Get files
    if pattern:
        candidates = glob_method(pattern)
    else:
        candidates = glob_method("*")

    # Apply filters
    for path in candidates:
        # File type filter
        if file_type:
            if file_type == 'file' and not path.is_file():
                continue
            elif file_type == 'dir' and not path.is_dir():
                continue
            elif file_type == 'symlink' and not path.is_symlink():
                continue

        # Skip directories unless specifically requested
        if file_type != 'dir' and path.is_dir():
            continue

        try:
            stat = path.stat()

            # Size filters
            if min_size is not None and stat.st_size < min_size:
                continue
            if max_size is not None and stat.st_size > max_size:
                continue

            # Time filters
            mtime = datetime.fromtimestamp(stat.st_mtime)
            if modified_after and mtime < modified_after:
                continue
            if modified_before and mtime > modified_before:
                continue

            yield path

        except OSError:
            # Skip files we can't stat
            continue


def copy_with_progress(
        src: Union[str, Path],
        dst: Union[str, Path],
        chunk_size: int = 1024 * 1024,
        progress_callback: Optional[Callable[[int, int], None]] = None
) -> Path:
    """
    Copy file with progress tracking.

    Args:
        src: Source file
        dst: Destination path
        chunk_size: Copy chunk size
        progress_callback: Callback(copied_bytes, total_bytes)

    Returns:
        Destination path

    Examples:
        >>> def progress(copied, total):
        ...     print(f"{copied}/{total} bytes ({copied/total*100:.1f}%)")
        >>> copy_with_progress("large.bin", "copy.bin", progress_callback=progress)
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    # Handle directory destination
    if dst.is_dir():
        dst = dst / src.name

    # Ensure destination directory exists
    ensure_directory(dst.parent)

    # Get file size
    total_size = src.stat().st_size
    copied = 0

    # Copy with progress
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break

            fdst.write(chunk)
            copied += len(chunk)

            if progress_callback:
                progress_callback(copied, total_size)

    # Copy metadata
    shutil.copystat(src, dst)

    return dst


def get_file_info(path: Union[str, Path]) -> FileInfo:
    """
    Get detailed file information.

    Args:
        path: File path

    Returns:
        FileInfo object

    Examples:
        >>> info = get_file_info("/tmp/data.txt")
        >>> print(f"Size: {info.size} bytes, Modified: {info.modified}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return FileInfo.from_path(path)


def watch_directory(
        directory: Union[str, Path],
        callback: Callable[[str, Path], None],
        events: List[str] = None,
        recursive: bool = True,
        poll_interval: float = 1.0
):
    """
    Watch directory for changes (simple polling implementation).

    Args:
        directory: Directory to watch
        callback: Callback(event_type, file_path)
        events: Event types to watch ['created', 'modified', 'deleted']
        recursive: Watch subdirectories
        poll_interval: Polling interval in seconds

    Note:
        This is a simple polling implementation.
        For production use, consider inotify on Linux or FSEvents on macOS.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    if events is None:
        events = ['created', 'modified', 'deleted']

    # Initial state
    file_states = {}

    def scan_directory():
        current_files = {}

        if recursive:
            paths = directory.rglob("*")
        else:
            paths = directory.glob("*")

        for path in paths:
            if path.is_file():
                try:
                    stat = path.stat()
                    current_files[str(path)] = stat.st_mtime
                except OSError:
                    pass

        return current_files

    # Initial scan
    file_states = scan_directory()

    # Watch loop
    try:
        while True:
            time.sleep(poll_interval)

            current_files = scan_directory()

            # Check for changes
            all_paths = set(file_states.keys()) | set(current_files.keys())

            for path_str in all_paths:
                path = Path(path_str)

                if path_str in current_files and path_str not in file_states:
                    # Created
                    if 'created' in events:
                        callback('created', path)

                elif path_str not in current_files and path_str in file_states:
                    # Deleted
                    if 'deleted' in events:
                        callback('deleted', path)

                elif path_str in current_files and path_str in file_states:
                    # Check if modified
                    if current_files[path_str] != file_states[path_str]:
                        if 'modified' in events:
                            callback('modified', path)

            # Update state
            file_states = current_files

    except KeyboardInterrupt:
        logger.info("Directory watching stopped")


class FileLock:
    """
    Simple file-based lock for process synchronization.

    Example:
        with FileLock("/tmp/myapp.lock"):
            # Exclusive access to resource
            process_data()
    """

    def __init__(
            self,
            path: Union[str, Path],
            timeout: Optional[float] = None,
            poll_interval: float = 0.1
    ):
        self.path = Path(path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.fd = None

        # Ensure lock directory exists
        ensure_directory(self.path.parent)

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the lock"""
        start_time = time.time()

        while True:
            try:
                # Open file
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR)

                # Try to acquire lock
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Write PID
                os.write(self.fd, f"{os.getpid()}\n".encode())

                return True

            except BlockingIOError:
                if not blocking:
                    return False

                # Check timeout
                if self.timeout is not None:
                    if time.time() - start_time > self.timeout:
                        return False

                time.sleep(self.poll_interval)

            except Exception as e:
                if self.fd is not None:
                    os.close(self.fd)
                    self.fd = None
                raise e

    def release(self):
        """Release the lock"""
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)
            except Exception:
                pass
            finally:
                self.fd = None

            # Remove lock file
            try:
                self.path.unlink()
            except OSError:
                pass

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire lock: {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


@contextmanager
def temporary_directory(
        prefix: str = "ops0_",
        cleanup: bool = True,
        base_dir: Optional[Union[str, Path]] = None
):
    """
    Context manager for temporary directory.

    Args:
        prefix: Directory name prefix
        cleanup: Remove directory on exit
        base_dir: Base directory for temp dir

    Examples:
        >>> with temporary_directory() as tmpdir:
        ...     (tmpdir / "data.txt").write_text("content")
    """
    if base_dir:
        base_dir = Path(base_dir)
        ensure_directory(base_dir)

    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=base_dir)
    tmpdir_path = Path(tmpdir)

    try:
        yield tmpdir_path
    finally:
        if cleanup and tmpdir_path.exists():
            shutil.rmtree(tmpdir_path)
"""
ops0 Network Utilities

Network operations, HTTP clients, and connectivity helpers.
"""

import socket
import time
import requests
import urllib.parse
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NetworkError(Exception):
    """Network-related errors"""
    pass


@dataclass
class DownloadProgress:
    """Download progress information"""
    url: str
    total_bytes: int
    downloaded_bytes: int
    elapsed_seconds: float

    @property
    def percentage(self) -> float:
        """Download percentage"""
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100

    @property
    def speed_mbps(self) -> float:
        """Download speed in MB/s"""
        if self.elapsed_seconds == 0:
            return 0.0
        return (self.downloaded_bytes / 1024 / 1024) / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining in seconds"""
        if self.downloaded_bytes == 0:
            return float('inf')

        rate = self.downloaded_bytes / self.elapsed_seconds
        remaining = self.total_bytes - self.downloaded_bytes
        return remaining / rate


def get_free_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """
    Find a free port starting from start_port.

    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        Free port number

    Raises:
        NetworkError: If no free port found
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue

    raise NetworkError(f"No free port found in range {start_port}-{start_port + max_attempts}")


def wait_for_port(
        host: str,
        port: int,
        timeout: float = 30.0,
        interval: float = 0.5
) -> bool:
    """
    Wait for a port to become available.

    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds

    Returns:
        True if port became available, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(interval)

    return False


def check_connectivity(url: str = "https://www.google.com", timeout: float = 5.0) -> bool:
    """
    Check internet connectivity.

    Args:
        url: URL to test connectivity
        timeout: Request timeout

    Returns:
        True if connected, False otherwise
    """
    try:
        response = requests.head(url, timeout=timeout)
        return response.status_code < 500
    except Exception:
        return False


def get_external_ip() -> Optional[str]:
    """
    Get external IP address.

    Returns:
        External IP address or None if failed
    """
    services = [
        "https://api.ipify.org",
        "https://icanhazip.com",
        "https://checkip.amazonaws.com"
    ]

    for service in services:
        try:
            response = requests.get(service, timeout=5)
            if response.status_code == 200:
                return response.text.strip()
        except Exception:
            continue

    return None


def check_internet_connection() -> bool:
    """
    Check if internet connection is available.

    Returns:
        True if connected to internet
    """
    # Try multiple methods
    if check_connectivity():
        return True

    # Try DNS resolution
    try:
        socket.gethostbyname("google.com")
        return True
    except socket.gaierror:
        pass

    # Try getting external IP
    if get_external_ip():
        return True

    return False


def download_with_progress(
        url: str,
        destination: Path,
        chunk_size: int = 8192,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        headers: Optional[Dict[str, str]] = None
) -> Path:
    """
    Download file with progress tracking.

    Args:
        url: URL to download from
        destination: Destination file path
        chunk_size: Download chunk size
        progress_callback: Optional callback for progress updates
        headers: Optional HTTP headers

    Returns:
        Path to downloaded file

    Raises:
        NetworkError: If download fails
    """
    try:
        start_time = time.time()

        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Start download
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()

        # Get total size
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        # Download with progress
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback:
                        progress = DownloadProgress(
                            url=url,
                            total_bytes=total_size,
                            downloaded_bytes=downloaded,
                            elapsed_seconds=time.time() - start_time
                        )
                        progress_callback(progress)

        logger.info(f"Downloaded {url} to {destination}")
        return destination

    except requests.RequestException as e:
        raise NetworkError(f"Failed to download {url}: {e}")


def upload_with_retry(
        url: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
) -> requests.Response:
    """
    Upload data with automatic retry on failure.

    Args:
        url: Upload URL
        data: Form data to upload
        files: Files to upload
        max_retries: Maximum number of retries
        backoff_factor: Exponential backoff factor
        timeout: Request timeout
        headers: Optional HTTP headers

    Returns:
        Response object

    Raises:
        NetworkError: If all retries fail
    """
    last_error = None
    delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                data=data,
                files=files,
                timeout=timeout,
                headers=headers
            )
            response.raise_for_status()
            return response

        except requests.RequestException as e:
            last_error = e

            if attempt < max_retries:
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}, retrying in {delay}s")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                break

    raise NetworkError(f"Failed to upload after {max_retries + 1} attempts: {last_error}")


class ConnectionPool:
    """
    HTTP connection pool for efficient request handling.

    Example:
        pool = ConnectionPool(max_connections=10)

        with pool.session() as session:
            response = session.get('https://api.example.com/data')
    """

    def __init__(
            self,
            max_connections: int = 10,
            max_retries: int = 3,
            timeout: float = 30.0
    ):
        self.max_connections = max_connections
        self.max_retries = max_retries
        self.timeout = timeout
        self._local = threading.local()

    def _get_session(self) -> requests.Session:
        """Get thread-local session"""
        if not hasattr(self._local, 'session'):
            session = requests.Session()

            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.max_connections,
                pool_maxsize=self.max_connections,
                max_retries=self.max_retries
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            # Set default timeout
            session.request = self._wrap_request(session.request)

            self._local.session = session

        return self._local.session

    def _wrap_request(self, original_request):
        """Wrap request method to add default timeout"""

        def wrapped(*args, **kwargs):
            if 'timeout' not in kwargs:
                kwargs['timeout'] = self.timeout
            return original_request(*args, **kwargs)

        return wrapped

    @contextmanager
    def session(self):
        """Get a session from the pool"""
        session = self._get_session()
        try:
            yield session
        except Exception:
            raise

    def close(self):
        """Close all sessions"""
        if hasattr(self._local, 'session'):
            self._local.session.close()
            del self._local.session


class HTTPClient:
    """
    High-level HTTP client with retries and error handling.

    Example:
        client = HTTPClient(base_url='https://api.example.com')

        # GET request
        data = client.get('/users/123')

        # POST request
        result = client.post('/users', json={'name': 'John'})

        # With custom headers
        client.headers['Authorization'] = 'Bearer token'
    """

    def __init__(
            self,
            base_url: Optional[str] = None,
            timeout: float = 30.0,
            max_retries: int = 3,
            backoff_factor: float = 2.0
    ):
        self.base_url = base_url.rstrip('/') if base_url else ''
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.headers: Dict[str, str] = {}
        self.session = requests.Session()

        # Configure retries
        retry_strategy = requests.adapters.Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _build_url(self, path: str) -> str:
        """Build full URL from path"""
        if path.startswith(('http://', 'https://')):
            return path

        path = path.lstrip('/')
        return f"{self.base_url}/{path}" if self.base_url else path

    def _request(
            self,
            method: str,
            path: str,
            **kwargs
    ) -> requests.Response:
        """Make HTTP request with error handling"""
        url = self._build_url(path)

        # Merge headers
        headers = self.headers.copy()
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        kwargs['headers'] = headers

        # Set timeout
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise NetworkError(f"{method} {url} failed: {e}")

    def get(self, path: str, **kwargs) -> Any:
        """GET request"""
        response = self._request('GET', path, **kwargs)
        return response.json() if response.content else None

    def post(self, path: str, **kwargs) -> Any:
        """POST request"""
        response = self._request('POST', path, **kwargs)
        return response.json() if response.content else None

    def put(self, path: str, **kwargs) -> Any:
        """PUT request"""
        response = self._request('PUT', path, **kwargs)
        return response.json() if response.content else None

    def delete(self, path: str, **kwargs) -> Any:
        """DELETE request"""
        response = self._request('DELETE', path, **kwargs)
        return response.json() if response.content else None

    def patch(self, path: str, **kwargs) -> Any:
        """PATCH request"""
        response = self._request('PATCH', path, **kwargs)
        return response.json() if response.content else None

    def close(self):
        """Close the session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
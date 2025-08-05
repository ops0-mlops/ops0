"""
ops0 Concurrency Utilities

Thread pools, async helpers, and synchronization primitives.
"""

import asyncio
import threading
import multiprocessing
import concurrent.futures
import functools
import time
import queue
from typing import Any, Callable, List, Optional, TypeVar, Union, Dict, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ThreadPoolManager:
    """
    Managed thread pool with monitoring and graceful shutdown.

    Example:
        pool = ThreadPoolManager(max_workers=4)

        # Submit tasks
        future = pool.submit(expensive_function, arg1, arg2)
        result = future.result()

        # Map function over items
        results = pool.map(process_item, items)

        # Graceful shutdown
        pool.shutdown()
    """

    def __init__(
            self,
            max_workers: Optional[int] = None,
            thread_name_prefix: str = "ops0-worker",
            monitor_interval: float = 60.0
    ):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.thread_name_prefix = thread_name_prefix
        self.monitor_interval = monitor_interval

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=self.thread_name_prefix
        )

        self._futures: List[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._shutdown = False
        self._stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'active': 0
        }

        # Start monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"{thread_name_prefix}-monitor",
            daemon=True
        )
        self._monitor_thread.start()

    def submit(
            self,
            fn: Callable[..., T],
            *args,
            **kwargs
    ) -> concurrent.futures.Future[T]:
        """Submit a function to be executed"""
        if self._shutdown:
            raise RuntimeError("Cannot submit to a shutdown pool")

        with self._lock:
            future = self.executor.submit(fn, *args, **kwargs)
            self._futures.append(future)
            self._stats['submitted'] += 1

            # Add completion callback
            future.add_done_callback(self._task_done)

            return future

    def map(
            self,
            fn: Callable[[T], Any],
            iterable,
            timeout: Optional[float] = None,
            chunksize: int = 1
    ) -> List[Any]:
        """Map function over iterable"""
        return list(self.executor.map(fn, iterable, timeout=timeout, chunksize=chunksize))

    def _task_done(self, future: concurrent.futures.Future):
        """Callback when task completes"""
        with self._lock:
            if future in self._futures:
                self._futures.remove(future)

            if future.exception():
                self._stats['failed'] += 1
            else:
                self._stats['completed'] += 1

    def _monitor_loop(self):
        """Monitor thread pool statistics"""
        while not self._shutdown:
            time.sleep(self.monitor_interval)

            with self._lock:
                self._stats['active'] = len(self._futures)

                # Clean up completed futures
                self._futures = [f for f in self._futures if not f.done()]

            logger.debug(
                f"ThreadPool stats: {self._stats['active']} active, "
                f"{self._stats['completed']} completed, "
                f"{self._stats['failed']} failed"
            )

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self._lock:
            return self._stats.copy()

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the thread pool"""
        self._shutdown = True
        self.executor.shutdown(wait=wait, timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)


def run_in_thread(
        func: Callable[..., T],
        *args,
        daemon: bool = True,
        name: Optional[str] = None,
        **kwargs
) -> threading.Thread:
    """
    Run function in a new thread.

    Args:
        func: Function to run
        *args: Positional arguments
        daemon: Run as daemon thread
        name: Thread name
        **kwargs: Keyword arguments

    Returns:
        Started thread

    Example:
        thread = run_in_thread(process_data, data, name="data-processor")
        thread.join()
    """
    thread = threading.Thread(
        target=func,
        args=args,
        kwargs=kwargs,
        daemon=daemon,
        name=name
    )
    thread.start()
    return thread


def run_async(coro_or_func, *args, **kwargs) -> Any:
    """
    Run async coroutine or sync function in event loop.

    Args:
        coro_or_func: Coroutine or async function
        *args: Arguments
        **kwargs: Keyword arguments

    Returns:
        Result of execution

    Example:
        async def fetch_data():
            await asyncio.sleep(1)
            return "data"

        result = run_async(fetch_data())
    """
    # Check if it's already a coroutine
    if asyncio.iscoroutine(coro_or_func):
        coro = coro_or_func
    # Check if it's an async function
    elif asyncio.iscoroutinefunction(coro_or_func):
        coro = coro_or_func(*args, **kwargs)
    else:
        # Regular function - run in executor
        loop = asyncio.new_event_loop()
        try:
            return loop.run_in_executor(None, coro_or_func, *args)
        finally:
            loop.close()

    # Run coroutine
    try:
        # Try to get existing event loop
        loop = asyncio.get_running_loop()
        # We're already in an async context
        task = asyncio.create_task(coro)
        return task
    except RuntimeError:
        # No event loop - create one
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def synchronized(lock: Optional[threading.Lock] = None):
    """
    Decorator to synchronize function/method access.

    Args:
        lock: Lock to use (creates new one if None)

    Example:
        class Counter:
            def __init__(self):
                self._count = 0
                self._lock = threading.Lock()

            @synchronized()
            def increment(self):
                self._count += 1
    """

    def decorator(func):
        # Use provided lock or create new one
        func_lock = lock or threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with func_lock:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    Example:
        # Allow 10 requests per second
        limiter = RateLimiter(rate=10, per=1.0)

        for i in range(20):
            with limiter:
                # This will be rate limited
                make_api_call()
    """

    def __init__(self, rate: float, per: float = 1.0, burst: Optional[int] = None):
        """
        Initialize rate limiter.

        Args:
            rate: Number of allowed events
            per: Time period in seconds
            burst: Maximum burst size (defaults to rate)
        """
        self.rate = rate
        self.per = per
        self.burst = burst or rate

        self._tokens = self.burst
        self._last_update = time.time()
        self._lock = threading.Lock()

    def _update_tokens(self):
        """Update available tokens"""
        now = time.time()
        elapsed = now - self._last_update

        # Add new tokens
        new_tokens = elapsed * (self.rate / self.per)
        self._tokens = min(self.burst, self._tokens + new_tokens)
        self._last_update = now

    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """
        Acquire tokens from rate limiter.

        Args:
            tokens: Number of tokens to acquire
            blocking: Wait if tokens not available

        Returns:
            True if acquired, False if not blocking and not available
        """
        with self._lock:
            while True:
                self._update_tokens()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                if not blocking:
                    return False

                # Calculate wait time
                deficit = tokens - self._tokens
                wait_time = deficit * (self.per / self.rate)

                # Release lock while waiting
                self._lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self._lock.acquire()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Semaphore:
    """
    Enhanced semaphore with timeout and statistics.

    Example:
        # Limit concurrent operations to 3
        sem = Semaphore(3)

        def process_item(item):
            with sem:
                # Only 3 concurrent executions
                expensive_operation(item)
    """

    def __init__(self, value: int = 1):
        self._semaphore = threading.Semaphore(value)
        self._max_value = value
        self._current_value = value
        self._lock = threading.Lock()
        self._stats = {
            'acquired': 0,
            'released': 0,
            'timeouts': 0,
            'max_wait_time': 0
        }

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire semaphore"""
        start_time = time.time()

        acquired = self._semaphore.acquire(blocking, timeout)

        with self._lock:
            if acquired:
                self._current_value -= 1
                self._stats['acquired'] += 1
                wait_time = time.time() - start_time
                self._stats['max_wait_time'] = max(
                    self._stats['max_wait_time'],
                    wait_time
                )
            else:
                self._stats['timeouts'] += 1

        return acquired

    def release(self):
        """Release semaphore"""
        self._semaphore.release()

        with self._lock:
            self._current_value += 1
            self._stats['released'] += 1

    def available(self) -> int:
        """Get number of available slots"""
        with self._lock:
            return self._current_value

    def get_stats(self) -> Dict[str, Any]:
        """Get semaphore statistics"""
        with self._lock:
            return {
                **self._stats,
                'available': self._current_value,
                'in_use': self._max_value - self._current_value
            }

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


async def async_retry(
        coro_func: Callable[..., Any],
        *args,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Exception, ...] = (Exception,),
        **kwargs
) -> Any:
    """
    Retry async function with exponential backoff.

    Args:
        coro_func: Async function to retry
        *args: Function arguments
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Example:
        result = await async_retry(
            fetch_data,
            url,
            max_attempts=5,
            delay=1.0
        )
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_attempts):
        try:
            return await coro_func(*args, **kwargs)
        except exceptions as e:
            last_exception = e

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                break

    raise last_exception


def parallel_map(
        func: Callable[[T], Any],
        items: List[T],
        max_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_threads: bool = True
) -> List[Any]:
    """
    Map function over items in parallel with progress tracking.

    Args:
        func: Function to apply
        items: Items to process
        max_workers: Maximum parallel workers
        chunk_size: Items per task
        progress_callback: Progress callback(completed, total)
        use_threads: Use threads (True) or processes (False)

    Returns:
        List of results

    Example:
        def process(item):
            return item ** 2

        results = parallel_map(
            process,
            range(1000),
            max_workers=4,
            progress_callback=lambda c, t: print(f"{c}/{t}")
        )
    """
    if not items:
        return []

    # Auto-detect chunk size
    if chunk_size is None:
        chunk_size = max(1, len(items) // (max_workers or multiprocessing.cpu_count() or 1))

    # Choose executor
    if use_threads:
        executor_class = concurrent.futures.ThreadPoolExecutor
    else:
        executor_class = concurrent.futures.ProcessPoolExecutor

    results = [None] * len(items)
    completed = 0
    lock = threading.Lock()

    def wrapped_func(idx_item):
        idx, item = idx_item
        result = func(item)

        with lock:
            nonlocal completed
            results[idx] = result
            completed += 1

            if progress_callback:
                progress_callback(completed, len(items))

        return idx, result

    # Process items
    with executor_class(max_workers=max_workers) as executor:
        # Submit tasks
        futures = []
        for i in range(0, len(items), chunk_size):
            chunk = [(j, items[j]) for j in range(i, min(i + chunk_size, len(items)))]

            for idx_item in chunk:
                future = executor.submit(wrapped_func, idx_item)
                futures.append(future)

        # Wait for completion
        concurrent.futures.wait(futures)

    return results


def create_worker_pool(
        worker_func: Callable[[queue.Queue, queue.Queue], None],
        num_workers: int,
        input_queue: Optional[queue.Queue] = None,
        output_queue: Optional[queue.Queue] = None,
        daemon: bool = True
) -> Tuple[List[threading.Thread], queue.Queue, queue.Queue]:
    """
    Create a pool of worker threads.

    Args:
        worker_func: Worker function(input_queue, output_queue)
        num_workers: Number of workers
        input_queue: Input queue (created if None)
        output_queue: Output queue (created if None)
        daemon: Create daemon threads

    Returns:
        Tuple of (workers, input_queue, output_queue)

    Example:
        def worker(in_q, out_q):
            while True:
                item = in_q.get()
                if item is None:
                    break
                result = process(item)
                out_q.put(result)

        workers, in_q, out_q = create_worker_pool(worker, 4)

        # Feed work
        for item in items:
            in_q.put(item)

        # Stop workers
        for _ in workers:
            in_q.put(None)
    """
    if input_queue is None:
        input_queue = queue.Queue()

    if output_queue is None:
        output_queue = queue.Queue()

    workers = []

    for i in range(num_workers):
        worker = threading.Thread(
            target=worker_func,
            args=(input_queue, output_queue),
            name=f"worker-{i}",
            daemon=daemon
        )
        worker.start()
        workers.append(worker)

    return workers, input_queue, output_queue


@contextmanager
def timeout_context(seconds: float):
    """
    Context manager for operations with timeout.

    Args:
        seconds: Timeout in seconds

    Example:
        with timeout_context(5.0):
            # Must complete within 5 seconds
            long_operation()
    """

    def timeout_handler():
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    timer = threading.Timer(seconds, timeout_handler)
    timer.start()

    try:
        yield
    finally:
        timer.cancel()
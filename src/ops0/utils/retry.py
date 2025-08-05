"""
ops0 Retry and Error Handling Utilities

Retry decorators, backoff strategies, and circuit breakers.
"""

import time
import random
import functools
import logging
from typing import Callable, Optional, Tuple, Type, Union, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retries"""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    RANDOM = "random"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, not allowing calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryError(Exception):
    """Raised when all retry attempts fail"""

    def __init__(self, message: str, last_exception: Optional[Exception] = None, attempts: int = 0):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_factor: float = 2.0
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
    jitter: bool = True
    jitter_range: float = 0.1

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        if self.backoff_strategy == BackoffStrategy.CONSTANT:
            delay = self.initial_delay

        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.initial_delay * attempt

        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))

        elif self.backoff_strategy == BackoffStrategy.FIBONACCI:
            # Fibonacci sequence
            a, b = self.initial_delay, self.initial_delay
            for _ in range(attempt - 1):
                a, b = b, a + b
            delay = a

        elif self.backoff_strategy == BackoffStrategy.RANDOM:
            delay = random.uniform(self.initial_delay, self.max_delay)

        else:
            delay = self.initial_delay

        # Apply max delay limit
        delay = min(delay, self.max_delay)

        # Add jitter
        if self.jitter and self.jitter_range > 0:
            jitter = delay * random.uniform(-self.jitter_range, self.jitter_range)
            delay = max(0, delay + jitter)

        return delay


def retry(
        max_attempts: Optional[int] = None,
        delay: Optional[float] = None,
        backoff: Optional[Union[float, BackoffStrategy]] = None,
        exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        on_retry: Optional[Callable[[Exception, int], None]] = None,
        on_failure: Optional[Callable[[Exception, int], None]] = None,
        config: Optional[RetryConfig] = None
):
    """
    Decorator for retrying functions with configurable backoff.

    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff factor or strategy
        exceptions: Exceptions to catch and retry
        on_retry: Callback on each retry(exception, attempt)
        on_failure: Callback on final failure(exception, attempts)
        config: Complete retry configuration

    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def flaky_api_call():
            response = requests.get("https://api.example.com")
            return response.json()

        # With callback
        @retry(
            max_attempts=5,
            on_retry=lambda e, n: logger.warning(f"Retry {n}: {e}")
        )
        def database_query():
            return db.execute("SELECT * FROM users")

        # With custom config
        config = RetryConfig(
            max_attempts=10,
            backoff_strategy=BackoffStrategy.FIBONACCI,
            jitter=True
        )
        @retry(config=config)
        def complex_operation():
            pass
    """
    # Build configuration
    if config is None:
        config = RetryConfig()

        if max_attempts is not None:
            config.max_attempts = max_attempts

        if delay is not None:
            config.initial_delay = delay

        if backoff is not None:
            if isinstance(backoff, BackoffStrategy):
                config.backoff_strategy = backoff
            else:
                config.backoff_factor = backoff

        if exceptions is not None:
            config.exceptions = exceptions

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except config.exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts:
                        # Final attempt failed
                        if on_failure:
                            on_failure(e, attempt)

                        raise RetryError(
                            f"{func.__name__} failed after {attempt} attempts",
                            last_exception=e,
                            attempts=attempt
                        )

                    # Calculate delay
                    delay = config.calculate_delay(attempt)

                    # Callback
                    if on_retry:
                        on_retry(e, attempt)

                    logger.debug(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {e.__class__.__name__}, waiting {delay:.2f}s"
                    )

                    time.sleep(delay)

        # Add retry control methods
        wrapper.retry_config = config

        return wrapper

    return decorator


def exponential_backoff(
        initial: float = 1.0,
        maximum: float = 60.0,
        factor: float = 2.0,
        jitter: bool = True
) -> Callable:
    """
    Create exponential backoff retry decorator.

    Args:
        initial: Initial delay
        maximum: Maximum delay
        factor: Exponential factor
        jitter: Add random jitter

    Example:
        @exponential_backoff(initial=0.5, maximum=30.0)
        def api_call():
            return requests.get("https://api.example.com")
    """
    config = RetryConfig(
        initial_delay=initial,
        max_delay=maximum,
        backoff_factor=factor,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        jitter=jitter
    )

    return retry(config=config)


def linear_backoff(
        initial: float = 1.0,
        maximum: float = 60.0,
        increment: float = 1.0
) -> Callable:
    """
    Create linear backoff retry decorator.

    Args:
        initial: Initial delay
        maximum: Maximum delay
        increment: Linear increment (multiplier)

    Example:
        @linear_backoff(initial=2.0, increment=2.0)
        def database_operation():
            return db.execute("SELECT * FROM large_table")
    """
    config = RetryConfig(
        initial_delay=initial,
        max_delay=maximum,
        backoff_factor=increment,
        backoff_strategy=BackoffStrategy.LINEAR
    )

    return retry(config=config)


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    state_changes: List[Tuple[CircuitState, datetime]] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Example:
        # Create circuit breaker
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=requests.RequestException
        )

        @breaker
        def call_external_service():
            response = requests.get("https://api.example.com")
            return response.json()

        # Or use as context manager
        with breaker:
            call_external_service()

        # Check state
        if breaker.current_state == CircuitState.OPEN:
            print("Service is down!")
    """

    def __init__(
            self,
            failure_threshold: int = 5,
            recovery_timeout: float = 60.0,
            expected_exception: Type[Exception] = Exception,
            success_threshold: int = 1,
            on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        self._half_open_successes = 0

    @property
    def current_state(self) -> CircuitState:
        """Get current circuit state"""
        with self._lock:
            return self._state

    def _change_state(self, new_state: CircuitState):
        """Change circuit state"""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes.append((new_state, datetime.now()))

        logger.info(f"Circuit breaker state changed: {old_state} -> {new_state}")

        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset circuit"""
        if self._state != CircuitState.OPEN:
            return False

        if self._stats.last_failure_time is None:
            return False

        return datetime.now() - self._stats.last_failure_time >= timedelta(seconds=self.recovery_timeout)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self._lock:
            # Check if circuit should attempt reset
            if self._should_attempt_reset():
                self._change_state(CircuitState.HALF_OPEN)
                self._half_open_successes = 0

            # Check circuit state
            if self._state == CircuitState.OPEN:
                raise Exception("Circuit breaker is OPEN")

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record success
                self._stats.total_calls += 1
                self._stats.successful_calls += 1
                self._stats.consecutive_failures = 0

                # Handle half-open state
                if self._state == CircuitState.HALF_OPEN:
                    self._half_open_successes += 1
                    if self._half_open_successes >= self.success_threshold:
                        self._change_state(CircuitState.CLOSED)

                return result

            except self.expected_exception as e:
                # Record failure
                self._stats.total_calls += 1
                self._stats.failed_calls += 1
                self._stats.consecutive_failures += 1
                self._stats.last_failure_time = datetime.now()

                # Check if we should open circuit
                if self._state == CircuitState.CLOSED:
                    if self._stats.consecutive_failures >= self.failure_threshold:
                        self._change_state(CircuitState.OPEN)

                elif self._state == CircuitState.HALF_OPEN:
                    # Failed in half-open state, go back to open
                    self._change_state(CircuitState.OPEN)

                raise

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                'state': self._state.value,
                'total_calls': self._stats.total_calls,
                'successful_calls': self._stats.successful_calls,
                'failed_calls': self._stats.failed_calls,
                'failure_rate': self._stats.failure_rate,
                'consecutive_failures': self._stats.consecutive_failures,
                'last_failure': self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None
            }

    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self._change_state(CircuitState.CLOSED)
            self._stats.consecutive_failures = 0
            self._half_open_successes = 0

    def __call__(self, func):
        """Use as decorator"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        wrapper.circuit_breaker = self
        return wrapper

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, self.expected_exception):
            # Exception was raised, circuit breaker already handled it
            return False


def fallback(
        fallback_func: Optional[Callable] = None,
        fallback_value: Any = None,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        on_fallback: Optional[Callable[[Exception], None]] = None
):
    """
    Decorator to provide fallback behavior on failure.

    Args:
        fallback_func: Function to call on failure
        fallback_value: Value to return on failure
        exceptions: Exceptions to catch
        on_fallback: Callback when fallback is used

    Example:
        @fallback(fallback_value="default")
        def get_config():
            return load_from_api()

        @fallback(fallback_func=lambda: get_from_cache())
        def get_data():
            return fetch_from_database()

        # With callback
        @fallback(
            fallback_value=[],
            on_fallback=lambda e: logger.error(f"Using fallback: {e}")
        )
        def get_users():
            return api.get_users()
    """
    if fallback_func is None and fallback_value is None:
        raise ValueError("Either fallback_func or fallback_value must be provided")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if on_fallback:
                    on_fallback(e)

                logger.debug(f"Using fallback for {func.__name__} due to {e.__class__.__name__}")

                if fallback_func:
                    return fallback_func()
                else:
                    return fallback_value

        return wrapper

    return decorator


# Convenience functions

def retry_on_exception(
        exception_type: Type[Exception],
        max_attempts: int = 3,
        delay: float = 1.0
) -> Callable:
    """
    Simple retry decorator for specific exception.

    Example:
        @retry_on_exception(ConnectionError, max_attempts=5)
        def connect_to_database():
            return db.connect()
    """
    return retry(
        max_attempts=max_attempts,
        delay=delay,
        exceptions=(exception_type,)
    )


def retry_with_timeout(
        timeout: float,
        max_attempts: int = 3,
        delay: float = 1.0
) -> Callable:
    """
    Retry with overall timeout limit.

    Example:
        @retry_with_timeout(timeout=30.0, max_attempts=10)
        def long_operation():
            return process_data()
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                # Check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"{func.__name__} timed out after {timeout}s "
                        f"and {attempt - 1} attempts"
                    )

                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_attempts:
                        remaining_time = timeout - (time.time() - start_time)
                        sleep_time = min(delay, remaining_time)

                        if sleep_time > 0:
                            time.sleep(sleep_time)

            raise RetryError(
                f"{func.__name__} failed after {max_attempts} attempts",
                last_exception=last_exception,
                attempts=max_attempts
            )

        return wrapper

    return decorator
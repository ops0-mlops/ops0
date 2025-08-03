"""
ops0 Hooks and Events System

Provides extensible hooks and events for customizing ops0 behavior
and integrating with external systems.
"""

import logging
import time
from typing import Dict, List, Callable, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from .exceptions import Ops0Error


class HookEvent(Enum):
    """Standard hook events in ops0 lifecycle"""

    # Pipeline events
    PIPELINE_CREATED = "pipeline.created"
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"

    # Step events
    STEP_CREATED = "step.created"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_RETRYING = "step.retrying"

    # Storage events
    STORAGE_SAVE = "storage.save"
    STORAGE_LOAD = "storage.load"
    STORAGE_DELETE = "storage.delete"

    # Deployment events
    DEPLOYMENT_STARTED = "deployment.started"
    DEPLOYMENT_COMPLETED = "deployment.completed"
    DEPLOYMENT_FAILED = "deployment.failed"

    # Monitoring events
    METRIC_RECORDED = "metric.recorded"
    ALERT_TRIGGERED = "alert.triggered"

    # Validation events
    VALIDATION_STARTED = "validation.started"
    VALIDATION_COMPLETED = "validation.completed"
    VALIDATION_FAILED = "validation.failed"


@dataclass
class HookContext:
    """Context information passed to hooks"""
    event: HookEvent
    timestamp: datetime
    pipeline_name: Optional[str] = None
    step_name: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get data value with default"""
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """Set data value"""
        self.data[key] = value


class Hook(ABC):
    """Abstract base class for hooks"""

    @abstractmethod
    def should_execute(self, context: HookContext) -> bool:
        """Determine if this hook should execute for the given context"""
        pass

    @abstractmethod
    def execute(self, context: HookContext) -> Optional[Any]:
        """Execute the hook logic"""
        pass

    @property
    def priority(self) -> int:
        """Hook execution priority (lower numbers execute first)"""
        return 100

    @property
    def name(self) -> str:
        """Hook name for identification"""
        return self.__class__.__name__


class FunctionHook(Hook):
    """Simple function-based hook"""

    def __init__(
            self,
            func: Callable[[HookContext], Any],
            events: List[HookEvent],
            name: Optional[str] = None,
            priority: int = 100,
            condition: Optional[Callable[[HookContext], bool]] = None
    ):
        self.func = func
        self.events = set(events)
        self._name = name or func.__name__
        self._priority = priority
        self.condition = condition

    def should_execute(self, context: HookContext) -> bool:
        if context.event not in self.events:
            return False

        if self.condition:
            return self.condition(context)

        return True

    def execute(self, context: HookContext) -> Optional[Any]:
        return self.func(context)

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def name(self) -> str:
        return self._name


class ConditionalHook(Hook):
    """Hook that executes only when certain conditions are met"""

    def __init__(
            self,
            hook: Hook,
            condition: Callable[[HookContext], bool],
            name: Optional[str] = None
    ):
        self.hook = hook
        self.condition = condition
        self._name = name or f"Conditional({hook.name})"

    def should_execute(self, context: HookContext) -> bool:
        return self.hook.should_execute(context) and self.condition(context)

    def execute(self, context: HookContext) -> Optional[Any]:
        return self.hook.execute(context)

    @property
    def priority(self) -> int:
        return self.hook.priority

    @property
    def name(self) -> str:
        return self._name


class TimingHook(Hook):
    """Hook that measures execution time of events"""

    def __init__(self, events: List[HookEvent], logger: Optional[logging.Logger] = None):
        self.events = set(events)
        self.logger = logger or logging.getLogger("ops0.timing")
        self.start_times: Dict[str, float] = {}

    def should_execute(self, context: HookContext) -> bool:
        return context.event in self.events

    def execute(self, context: HookContext) -> Optional[Any]:
        event_key = f"{context.pipeline_name}:{context.step_name}" if context.step_name else context.pipeline_name

        if context.event.value.endswith('.started'):
            self.start_times[event_key] = time.time()
        elif context.event.value.endswith('.completed') or context.event.value.endswith('.failed'):
            if event_key in self.start_times:
                duration = time.time() - self.start_times[event_key]
                self.logger.info(f"⏱️  {context.event.value}: {event_key} took {duration:.3f}s")
                del self.start_times[event_key]
                return duration

        return None


class LoggingHook(Hook):
    """Hook that logs events with structured information"""

    def __init__(
            self,
            events: List[HookEvent],
            logger: Optional[logging.Logger] = None,
            log_level: int = logging.INFO,
            include_data: bool = False
    ):
        self.events = set(events)
        self.logger = logger or logging.getLogger("ops0.hooks")
        self.log_level = log_level
        self.include_data = include_data

    def should_execute(self, context: HookContext) -> bool:
        return context.event in self.events

    def execute(self, context: HookContext) -> Optional[Any]:
        message_parts = [f"🔗 {context.event.value}"]

        if context.pipeline_name:
            message_parts.append(f"pipeline:{context.pipeline_name}")

        if context.step_name:
            message_parts.append(f"step:{context.step_name}")

        if self.include_data and context.data:
            message_parts.append(f"data:{context.data}")

        message = " | ".join(message_parts)
        self.logger.log(self.log_level, message)


class MetricsHook(Hook):
    """Hook that collects metrics from events"""

    def __init__(self, events: List[HookEvent]):
        self.events = set(events)
        self.metrics: Dict[str, Any] = {
            "event_counts": {},
            "execution_times": {},
            "error_counts": {},
            "last_execution": {}
        }

    def should_execute(self, context: HookContext) -> bool:
        return context.event in self.events

    def execute(self, context: HookContext) -> Optional[Any]:
        event_name = context.event.value

        # Count events
        self.metrics["event_counts"][event_name] = self.metrics["event_counts"].get(event_name, 0) + 1

        # Track last execution
        self.metrics["last_execution"][event_name] = context.timestamp

        # Track errors
        if event_name.endswith('.failed'):
            self.metrics["error_counts"][event_name] = self.metrics["error_counts"].get(event_name, 0) + 1

        # Track execution times if available
        if "execution_time" in context.data:
            if event_name not in self.metrics["execution_times"]:
                self.metrics["execution_times"][event_name] = []
            self.metrics["execution_times"][event_name].append(context.data["execution_time"])

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset all collected metrics"""
        self.metrics = {
            "event_counts": {},
            "execution_times": {},
            "error_counts": {},
            "last_execution": {}
        }


class AlertHook(Hook):
    """Hook that triggers alerts based on conditions"""

    def __init__(
            self,
            events: List[HookEvent],
            alert_condition: Callable[[HookContext], bool],
            alert_handler: Callable[[HookContext], None],
            name: str = "AlertHook"
    ):
        self.events = set(events)
        self.alert_condition = alert_condition
        self.alert_handler = alert_handler
        self._name = name

    def should_execute(self, context: HookContext) -> bool:
        return context.event in self.events and self.alert_condition(context)

    def execute(self, context: HookContext) -> Optional[Any]:
        try:
            self.alert_handler(context)
        except Exception as e:
            logging.getLogger("ops0.alerts").error(f"Alert handler failed: {e}")

    @property
    def name(self) -> str:
        return self._name


class HookManager:
    """Central manager for hooks and events"""

    def __init__(self):
        self.hooks: Dict[HookEvent, List[Hook]] = {}
        self.global_hooks: List[Hook] = []  # Hooks that listen to all events
        self._logger = logging.getLogger("ops0.hooks")
        self._enabled = True

    def register_hook(self, hook: Hook, events: Optional[List[HookEvent]] = None):
        """
        Register a hook for specific events or globally.

        Args:
            hook: Hook to register
            events: Events to listen to (None for all events)
        """
        if events is None:
            self.global_hooks.append(hook)
            self.global_hooks.sort(key=lambda h: h.priority)
            self._logger.debug(f"Registered global hook: {hook.name}")
        else:
            for event in events:
                if event not in self.hooks:
                    self.hooks[event] = []
                self.hooks[event].append(hook)
                self.hooks[event].sort(key=lambda h: h.priority)
            self._logger.debug(f"Registered hook '{hook.name}' for events: {[e.value for e in events]}")

    def unregister_hook(self, hook: Hook):
        """Unregister a hook from all events"""
        # Remove from global hooks
        if hook in self.global_hooks:
            self.global_hooks.remove(hook)

        # Remove from specific event hooks
        for event_hooks in self.hooks.values():
            if hook in event_hooks:
                event_hooks.remove(hook)

        self._logger.debug(f"Unregistered hook: {hook.name}")

    def trigger_event(
            self,
            event: HookEvent,
            pipeline_name: Optional[str] = None,
            step_name: Optional[str] = None,
            data: Optional[Dict[str, Any]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Trigger an event and execute all applicable hooks.

        Args:
            event: Event to trigger
            pipeline_name: Pipeline name context
            step_name: Step name context
            data: Event data
            metadata: Event metadata

        Returns:
            List of hook execution results
        """
        if not self._enabled:
            return []

        context = HookContext(
            event=event,
            timestamp=datetime.now(),
            pipeline_name=pipeline_name,
            step_name=step_name,
            data=data or {},
            metadata=metadata or {}
        )

        results = []

        # Execute global hooks
        for hook in self.global_hooks:
            if hook.should_execute(context):
                try:
                    result = hook.execute(context)
                    results.append(result)
                except Exception as e:
                    self._logger.error(f"Global hook '{hook.name}' failed: {e}")

        # Execute event-specific hooks
        if event in self.hooks:
            for hook in self.hooks[event]:
                if hook.should_execute(context):
                    try:
                        result = hook.execute(context)
                        results.append(result)
                    except Exception as e:
                        self._logger.error(f"Hook '{hook.name}' for event '{event.value}' failed: {e}")

        return results

    def enable(self):
        """Enable hook execution"""
        self._enabled = True
        self._logger.debug("Hook execution enabled")

    def disable(self):
        """Disable hook execution"""
        self._enabled = False
        self._logger.debug("Hook execution disabled")

    def is_enabled(self) -> bool:
        """Check if hook execution is enabled"""
        return self._enabled

    def list_hooks(self) -> Dict[str, List[str]]:
        """List all registered hooks by event"""
        result = {}

        if self.global_hooks:
            result["global"] = [hook.name for hook in self.global_hooks]

        for event, hooks in self.hooks.items():
            if hooks:
                result[event.value] = [hook.name for hook in hooks]

        return result

    def clear_hooks(self):
        """Remove all registered hooks"""
        self.hooks.clear()
        self.global_hooks.clear()
        self._logger.debug("All hooks cleared")


# Global hook manager instance
hook_manager = HookManager()


# Convenience functions for common hook patterns
def on_event(events: List[HookEvent], priority: int = 100):
    """
    Decorator to register a function as a hook for specific events.

    Example:
        @ops0.on_event([HookEvent.STEP_COMPLETED])
        def log_step_completion(context):
            print(f"Step {context.step_name} completed!")
    """

    def decorator(func: Callable[[HookContext], Any]) -> Callable[[HookContext], Any]:
        hook = FunctionHook(func, events, priority=priority)
        hook_manager.register_hook(hook, events)
        return func

    return decorator


def on_pipeline_event(events: List[str]):
    """
    Decorator for pipeline-specific events.

    Example:
        @ops0.on_pipeline_event(['started', 'completed'])
        def track_pipeline(context):
            # Track pipeline lifecycle
            pass
    """
    hook_events = []
    for event_name in events:
        if hasattr(HookEvent, f"PIPELINE_{event_name.upper()}"):
            hook_events.append(getattr(HookEvent, f"PIPELINE_{event_name.upper()}"))

    return on_event(hook_events)


def on_step_event(events: List[str]):
    """
    Decorator for step-specific events.

    Example:
        @ops0.on_step_event(['failed'])
        def alert_on_failure(context):
            send_alert(f"Step {context.step_name} failed!")
    """
    hook_events = []
    for event_name in events:
        if hasattr(HookEvent, f"STEP_{event_name.upper()}"):
            hook_events.append(getattr(HookEvent, f"STEP_{event_name.upper()}"))

    return on_event(hook_events)


# Pre-built hook instances for common use cases
def create_timing_hook() -> TimingHook:
    """Create a timing hook for all start/complete events"""
    timing_events = [
        HookEvent.PIPELINE_STARTED, HookEvent.PIPELINE_COMPLETED,
        HookEvent.STEP_STARTED, HookEvent.STEP_COMPLETED,
        HookEvent.DEPLOYMENT_STARTED, HookEvent.DEPLOYMENT_COMPLETED
    ]
    return TimingHook(timing_events)


def create_logging_hook(log_level: int = logging.INFO) -> LoggingHook:
    """Create a logging hook for all events"""
    return LoggingHook(list(HookEvent), log_level=log_level)


def create_metrics_hook() -> MetricsHook:
    """Create a metrics collection hook"""
    return MetricsHook(list(HookEvent))


def create_failure_alert_hook(alert_handler: Callable[[HookContext], None]) -> AlertHook:
    """Create an alert hook for failure events"""
    failure_events = [
        HookEvent.PIPELINE_FAILED,
        HookEvent.STEP_FAILED,
        HookEvent.DEPLOYMENT_FAILED,
        HookEvent.VALIDATION_FAILED
    ]

    def failure_condition(context: HookContext) -> bool:
        return True  # Alert on all failures

    return AlertHook(failure_events, failure_condition, alert_handler, "FailureAlertHook")


# Auto-register default hooks in development mode
def _setup_default_hooks():
    """Setup default hooks for development"""
    import os

    if os.getenv("OPS0_ENV", "").lower() == "development":
        # Register timing hook
        timing_hook = create_timing_hook()
        hook_manager.register_hook(timing_hook)

        # Register basic logging hook
        logging_hook = create_logging_hook(logging.DEBUG)
        hook_manager.register_hook(logging_hook)


# Initialize default hooks
_setup_default_hooks()
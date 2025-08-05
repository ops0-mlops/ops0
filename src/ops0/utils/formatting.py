"""
ops0 Formatting Utilities

Human-readable formatting for various data types.
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import difflib
import textwrap


class Color(Enum):
    """ANSI color codes"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'

    # Colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


def format_bytes(size_bytes: int, precision: int = 2) -> str:
    """
    Format byte size in human-readable format.

    Args:
        size_bytes: Size in bytes
        precision: Decimal precision

    Returns:
        Formatted string (e.g., "1.23 GB")

    Examples:
        >>> format_bytes(1024)
        '1.00 KB'
        >>> format_bytes(1234567890)
        '1.15 GB'
    """
    if size_bytes == 0:
        return "0 B"

    # Handle negative sizes
    sign = '-' if size_bytes < 0 else ''
    size_bytes = abs(size_bytes)

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{sign}{size:.{precision}f} {units[unit_index]}"


def format_duration(seconds: float, precision: str = 'auto') -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds
        precision: 'auto', 'seconds', 'minutes', 'hours', or 'full'

    Returns:
        Formatted duration string

    Examples:
        >>> format_duration(123.45)
        '2m 3s'
        >>> format_duration(3661)
        '1h 1m 1s'
        >>> format_duration(0.123)
        '123ms'
    """
    if seconds < 0:
        return f"-{format_duration(abs(seconds), precision)}"

    # Handle sub-second durations
    if seconds < 1 and precision in ['auto', 'seconds']:
        if seconds < 0.001:
            return f"{seconds * 1_000_000:.0f}μs"
        elif seconds < 1:
            return f"{seconds * 1000:.0f}ms"

    # Calculate components
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    parts = []

    if precision == 'full' or (precision == 'auto' and days > 0):
        if days > 0:
            parts.append(f"{days}d")

    if precision in ['full', 'hours'] or (precision == 'auto' and (hours > 0 or days > 0)):
        if hours > 0 or (precision == 'full' and days > 0):
            parts.append(f"{hours}h")

    if precision in ['full', 'hours', 'minutes'] or (precision == 'auto' and (minutes > 0 or hours > 0 or days > 0)):
        if minutes > 0 or (precision != 'auto' and (hours > 0 or days > 0)):
            parts.append(f"{minutes}m")

    if precision in ['full', 'seconds'] or (precision == 'auto' and (secs > 0 or not parts)):
        if secs >= 1:
            parts.append(f"{int(secs)}s")
        elif not parts:  # Show decimal seconds only if no larger units
            parts.append(f"{secs:.1f}s")

    return ' '.join(parts) if parts else '0s'


def format_timestamp(
        timestamp: Optional[Union[datetime, float]] = None,
        format: str = 'iso'
) -> str:
    """
    Format timestamp in various formats.

    Args:
        timestamp: Datetime or Unix timestamp (defaults to now)
        format: 'iso', 'human', 'relative', 'date', 'time', or custom strftime

    Returns:
        Formatted timestamp string

    Examples:
        >>> format_timestamp(datetime(2024, 1, 1, 12, 0, 0), 'human')
        '2024-01-01 12:00:00'
        >>> format_timestamp(time.time() - 3600, 'relative')
        '1 hour ago'
    """
    # Convert to datetime
    if timestamp is None:
        dt = datetime.now()
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    else:
        dt = timestamp

    # Format based on type
    if format == 'iso':
        return dt.isoformat()
    elif format == 'human':
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    elif format == 'date':
        return dt.strftime('%Y-%m-%d')
    elif format == 'time':
        return dt.strftime('%H:%M:%S')
    elif format == 'relative':
        return _format_relative_time(dt)
    else:
        # Custom format
        return dt.strftime(format)


def _format_relative_time(dt: datetime) -> str:
    """Format datetime as relative time"""
    now = datetime.now()
    delta = now - dt

    if delta.total_seconds() < 0:
        # Future time
        delta = dt - now
        suffix = "from now"
    else:
        suffix = "ago"

    seconds = int(delta.total_seconds())

    if seconds < 60:
        return f"{seconds} seconds {suffix}"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} {suffix}"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} {suffix}"
    elif seconds < 604800:
        days = seconds // 86400
        return f"{days} day{'s' if days != 1 else ''} {suffix}"
    elif seconds < 2592000:
        weeks = seconds // 604800
        return f"{weeks} week{'s' if weeks != 1 else ''} {suffix}"
    elif seconds < 31536000:
        months = seconds // 2592000
        return f"{months} month{'s' if months != 1 else ''} {suffix}"
    else:
        years = seconds // 31536000
        return f"{years} year{'s' if years != 1 else ''} {suffix}"


def format_progress(
        current: int,
        total: int,
        width: int = 50,
        fill_char: str = '█',
        empty_char: str = '░'
) -> str:
    """
    Format progress bar.

    Args:
        current: Current value
        total: Total value
        width: Bar width in characters
        fill_char: Character for filled portion
        empty_char: Character for empty portion

    Returns:
        Progress bar string

    Examples:
        >>> format_progress(30, 100, width=20)
        '██████░░░░░░░░░░░░░░ 30%'
    """
    if total == 0:
        percentage = 0
    else:
        percentage = min(100, max(0, int(100 * current / total)))

    filled_width = int(width * percentage / 100)
    empty_width = width - filled_width

    bar = fill_char * filled_width + empty_char * empty_width
    return f"{bar} {percentage}%"


def format_table(
        data: List[Dict[str, Any]],
        headers: Optional[List[str]] = None,
        max_width: Optional[int] = None,
        align: Dict[str, str] = None
) -> str:
    """
    Format data as ASCII table.

    Args:
        data: List of dictionaries
        headers: Column headers (auto-detected if None)
        max_width: Maximum column width
        align: Column alignment ('left', 'right', 'center')

    Returns:
        Formatted table string

    Examples:
        >>> data = [
        ...     {'name': 'Alice', 'age': 30, 'city': 'New York'},
        ...     {'name': 'Bob', 'age': 25, 'city': 'Los Angeles'}
        ... ]
        >>> print(format_table(data))
        ┌───────┬─────┬─────────────┐
        │ name  │ age │ city        │
        ├───────┼─────┼─────────────┤
        │ Alice │  30 │ New York    │
        │ Bob   │  25 │ Los Angeles │
        └───────┴─────┴─────────────┘
    """
    if not data:
        return "No data"

    # Auto-detect headers
    if headers is None:
        headers = list(data[0].keys())

    # Calculate column widths
    widths = {}
    for header in headers:
        widths[header] = len(str(header))
        for row in data:
            value = str(row.get(header, ''))
            widths[header] = max(widths[header], len(value))

    # Apply max width
    if max_width:
        for header in headers:
            widths[header] = min(widths[header], max_width)

    # Default alignment
    if align is None:
        align = {}

    # Build table
    lines = []

    # Top border
    top_parts = []
    for header in headers:
        top_parts.append('─' * (widths[header] + 2))
    lines.append('┌' + '┬'.join(top_parts) + '┐')

    # Header row
    header_parts = []
    for header in headers:
        value = str(header)
        if len(value) > widths[header]:
            value = value[:widths[header] - 3] + '...'

        alignment = align.get(header, 'left')
        if alignment == 'right':
            value = value.rjust(widths[header])
        elif alignment == 'center':
            value = value.center(widths[header])
        else:
            value = value.ljust(widths[header])

        header_parts.append(f" {value} ")
    lines.append('│' + '│'.join(header_parts) + '│')

    # Header separator
    sep_parts = []
    for header in headers:
        sep_parts.append('─' * (widths[header] + 2))
    lines.append('├' + '┼'.join(sep_parts) + '┤')

    # Data rows
    for row in data:
        row_parts = []
        for header in headers:
            value = str(row.get(header, ''))
            if len(value) > widths[header]:
                value = value[:widths[header] - 3] + '...'

            alignment = align.get(header, 'left')
            if alignment == 'right':
                value = value.rjust(widths[header])
            elif alignment == 'center':
                value = value.center(widths[header])
            else:
                value = value.ljust(widths[header])

            row_parts.append(f" {value} ")
        lines.append('│' + '│'.join(row_parts) + '│')

    # Bottom border
    bottom_parts = []
    for header in headers:
        bottom_parts.append('─' * (widths[header] + 2))
    lines.append('└' + '┴'.join(bottom_parts) + '┘')

    return '\n'.join(lines)


def format_diff(
        old_text: str,
        new_text: str,
        context_lines: int = 3,
        color: bool = True
) -> str:
    """
    Format text diff in unified format.

    Args:
        old_text: Original text
        new_text: Modified text
        context_lines: Number of context lines
        color: Add color codes

    Returns:
        Formatted diff string
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile='old',
        tofile='new',
        n=context_lines
    )

    if not color:
        return ''.join(diff)

    # Add colors
    colored_lines = []
    for line in diff:
        if line.startswith('+++') or line.startswith('---'):
            colored_lines.append(f"{Color.BOLD.value}{line}{Color.RESET.value}")
        elif line.startswith('+'):
            colored_lines.append(f"{Color.GREEN.value}{line}{Color.RESET.value}")
        elif line.startswith('-'):
            colored_lines.append(f"{Color.RED.value}{line}{Color.RESET.value}")
        elif line.startswith('@'):
            colored_lines.append(f"{Color.CYAN.value}{line}{Color.RESET.value}")
        else:
            colored_lines.append(line)

    return ''.join(colored_lines)


def truncate_string(
        text: str,
        max_length: int,
        suffix: str = '...',
        whole_words: bool = True
) -> str:
    """
    Truncate string to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        whole_words: Truncate at word boundaries

    Returns:
        Truncated string

    Examples:
        >>> truncate_string("This is a long text", 10)
        'This is...'
    """
    if len(text) <= max_length:
        return text

    max_length -= len(suffix)

    if whole_words:
        # Find last space before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > 0:
            truncated = truncated[:last_space]
    else:
        truncated = text[:max_length]

    return truncated + suffix


def humanize_number(
        number: Union[int, float],
        precision: int = 2,
        thousand_sep: str = ','
) -> str:
    """
    Format number in human-readable format.

    Args:
        number: Number to format
        precision: Decimal precision
        thousand_sep: Thousand separator

    Returns:
        Formatted number string

    Examples:
        >>> humanize_number(1234567.89)
        '1,234,567.89'
        >>> humanize_number(1234567890)
        '1.23B'
    """
    # Handle special cases
    if math.isnan(number):
        return 'NaN'
    elif math.isinf(number):
        return '∞' if number > 0 else '-∞'

    # For very large numbers, use suffixes
    if abs(number) >= 1e9:
        suffixes = ['', 'K', 'M', 'B', 'T', 'P']
        suffix_index = 0
        num = float(number)

        while abs(num) >= 1000 and suffix_index < len(suffixes) - 1:
            num /= 1000
            suffix_index += 1

        return f"{num:.{precision}f}{suffixes[suffix_index]}"

    # For regular numbers, add thousand separators
    if isinstance(number, int):
        formatted = f"{number:,}".replace(',', thousand_sep)
    else:
        formatted = f"{number:,.{precision}f}".replace(',', thousand_sep)

    return formatted


def colorize_text(
        text: str,
        color: Union[str, Color],
        bold: bool = False,
        underline: bool = False
) -> str:
    """
    Add ANSI color codes to text.

    Args:
        text: Text to colorize
        color: Color name or Color enum
        bold: Make text bold
        underline: Underline text

    Returns:
        Colorized text

    Examples:
        >>> print(colorize_text("Hello", "red", bold=True))
        <red bold text>
    """
    # Convert string color to Color enum
    if isinstance(color, str):
        color_map = {
            'black': Color.BLACK,
            'red': Color.RED,
            'green': Color.GREEN,
            'yellow': Color.YELLOW,
            'blue': Color.BLUE,
            'magenta': Color.MAGENTA,
            'cyan': Color.CYAN,
            'white': Color.WHITE,
        }
        color = color_map.get(color.lower(), Color.RESET)

    # Build color string
    codes = []
    if bold:
        codes.append(Color.BOLD.value)
    if underline:
        codes.append(Color.UNDERLINE.value)
    codes.append(color.value)

    prefix = ''.join(codes)
    return f"{prefix}{text}{Color.RESET.value}"


def create_progress_bar(
        current: int,
        total: int,
        prefix: str = '',
        suffix: str = '',
        width: int = 50,
        fill: str = '█',
        empty: str = '░',
        show_percentage: bool = True,
        show_count: bool = True,
        show_time: bool = False,
        elapsed_time: Optional[float] = None
) -> str:
    """
    Create a detailed progress bar.

    Args:
        current: Current value
        total: Total value
        prefix: Text before bar
        suffix: Text after bar
        width: Bar width
        fill: Fill character
        empty: Empty character
        show_percentage: Show percentage
        show_count: Show count (current/total)
        show_time: Show elapsed/remaining time
        elapsed_time: Elapsed time in seconds

    Returns:
        Progress bar string

    Examples:
        >>> create_progress_bar(30, 100, prefix='Processing: ')
        'Processing: ██████████████░░░░░░░░░░░░░░░░░░░░ 30% (30/100)'
    """
    parts = []

    if prefix:
        parts.append(prefix)

    # Progress bar
    percentage = 0 if total == 0 else min(100, max(0, int(100 * current / total)))
    filled_width = int(width * percentage / 100)
    bar = fill * filled_width + empty * (width - filled_width)
    parts.append(bar)

    # Statistics
    stats = []

    if show_percentage:
        stats.append(f"{percentage}%")

    if show_count:
        stats.append(f"{current}/{total}")

    if show_time and elapsed_time is not None:
        elapsed_str = format_duration(elapsed_time)
        stats.append(f"[{elapsed_str}")

        # Estimate remaining time
        if current > 0 and current < total:
            rate = current / elapsed_time
            remaining = (total - current) / rate
            remaining_str = format_duration(remaining)
            stats.append(f"ETA {remaining_str}")

        stats[-1] += "]"

    if stats:
        parts.append(f" {' '.join(stats)}")

    if suffix:
        parts.append(suffix)

    return ''.join(parts)
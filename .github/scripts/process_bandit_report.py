#!/usr/bin/env python3
"""
Process Bandit security scan reports and format them for GitHub Actions.

This script reads Bandit JSON reports and generates:
- Markdown summaries for PR comments
- Security metrics for monitoring
- Filtered results by severity level
- Actionable recommendations for developers

Usage:
    python process_bandit_report.py bandit-report.json
    python process_bandit_report.py bandit-report.json --format markdown
    python process_bandit_report.py bandit-report.json --severity high,medium
    python process_bandit_report.py bandit-report.json --output summary.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BanditIssue:
    """Represents a single Bandit security issue."""
    test_id: str
    test_name: str
    filename: str
    line_number: int
    code: str
    severity: str
    confidence: str
    issue_text: str
    more_info: str


class BanditReportProcessor:
    """Process and format Bandit security scan reports."""

    SEVERITY_LEVELS = {
        'LOW': 1,
        'MEDIUM': 2,
        'HIGH': 3
    }

    CONFIDENCE_LEVELS = {
        'LOW': 1,
        'MEDIUM': 2,
        'HIGH': 3
    }

    # ops0-specific security patterns to highlight
    OPS0_CRITICAL_PATTERNS = {
        'B102': 'exec_used',  # Use of exec
        'B301': 'pickle',  # Pickle usage (important for ML models)
        'B506': 'yaml_load',  # Unsafe YAML loading
        'B603': 'subprocess_without_shell_equals_true',  # Subprocess usage
        'B608': 'hardcoded_sql_expressions',  # SQL injection
        'B201': 'flask_debug_true',  # Flask debug mode
    }

    def __init__(self, report_path: str):
        self.report_path = Path(report_path)
        self.issues: List[BanditIssue] = []
        self.metrics: Dict = {}

    def load_report(self) -> bool:
        """Load and parse the Bandit JSON report."""
        try:
            if not self.report_path.exists():
                print(f"Error: Report file not found: {self.report_path}")
                return False

            with open(self.report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Parse Bandit report structure
            if 'results' not in data:
                print("Warning: No 'results' section found in Bandit report")
                return True

            for result in data['results']:
                issue = BanditIssue(
                    test_id=result.get('test_id', 'Unknown'),
                    test_name=result.get('test_name', 'Unknown Test'),
                    filename=result.get('filename', 'Unknown'),
                    line_number=result.get('line_number', 0),
                    code=result.get('code', '').strip(),
                    severity=result.get('issue_severity', 'UNDEFINED').upper(),
                    confidence=result.get('issue_confidence', 'UNDEFINED').upper(),
                    issue_text=result.get('issue_text', ''),
                    more_info=result.get('more_info', '')
                )
                self.issues.append(issue)

            # Store additional metrics from the report
            self.metrics = data.get('metrics', {})

            return True

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in report file: {e}")
            return False
        except Exception as e:
            print(f"Error loading report: {e}")
            return False

    def filter_issues(self,
                      severity_filter: Optional[Set[str]] = None,
                      confidence_filter: Optional[Set[str]] = None,
                      exclude_files: Optional[Set[str]] = None) -> List[BanditIssue]:
        """Filter issues based on criteria."""
        filtered_issues = []

        for issue in self.issues:
            # Filter by severity
            if severity_filter and issue.severity not in severity_filter:
                continue

            # Filter by confidence
            if confidence_filter and issue.confidence not in confidence_filter:
                continue

            # Filter by file patterns
            if exclude_files:
                skip = False
                for pattern in exclude_files:
                    if pattern in issue.filename:
                        skip = True
                        break
                if skip:
                    continue

            filtered_issues.append(issue)

        return filtered_issues

    def get_severity_summary(self) -> Dict[str, int]:
        """Get count of issues by severity level."""
        summary = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNDEFINED': 0}

        for issue in self.issues:
            severity = issue.severity.upper()
            if severity in summary:
                summary[severity] += 1
            else:
                summary['UNDEFINED'] += 1

        return summary

    def get_critical_ops0_issues(self) -> List[BanditIssue]:
        """Get issues that are critical for ops0 MLOps workflows."""
        critical_issues = []

        for issue in self.issues:
            if issue.test_id in self.OPS0_CRITICAL_PATTERNS:
                critical_issues.append(issue)

        return critical_issues

    def generate_markdown_summary(self,
                                  severity_filter: Optional[Set[str]] = None,
                                  include_code: bool = True) -> str:
        """Generate a markdown summary of security issues."""
        filtered_issues = self.filter_issues(severity_filter=severity_filter)
        severity_summary = self.get_severity_summary()
        critical_issues = self.get_critical_ops0_issues()

        # Build markdown report
        lines = []
        lines.append("## 🔒 Bandit Security Scan Results")
        lines.append("")

        # Executive summary
        total_issues = len(self.issues)
        if total_issues == 0:
            lines.append("✅ **No security issues found!**")
            return "\n".join(lines)

        lines.append(f"📊 **Total Issues**: {total_issues}")
        lines.append("")

        # Severity breakdown
        lines.append("### 📈 Issue Breakdown by Severity")
        lines.append("")

        for severity, count in severity_summary.items():
            if count > 0:
                emoji = self._get_severity_emoji(severity)
                lines.append(f"- {emoji} **{severity}**: {count} issues")
        lines.append("")

        # Critical ops0-specific issues
        if critical_issues:
            lines.append("### 🚨 Critical Issues for MLOps Workflows")
            lines.append("")
            lines.append("These issues are particularly important for ops0 deployments:")
            lines.append("")

            for issue in critical_issues:
                pattern_name = self.OPS0_CRITICAL_PATTERNS.get(issue.test_id, 'security_issue')
                lines.append(f"- **{issue.test_name}** (`{issue.test_id}`)")
                lines.append(f"  - 📁 File: `{issue.filename}:{issue.line_number}`")
                lines.append(f"  - ⚠️ Severity: {issue.severity} | Confidence: {issue.confidence}")
                lines.append(f"  - 📝 Issue: {issue.issue_text}")
                if issue.more_info:
                    lines.append(f"  - 🔗 More info: {issue.more_info}")
                lines.append("")

        # Detailed issues (if not too many)
        if len(filtered_issues) <= 20:
            lines.append(f"### 📋 Detailed Issues")
            if severity_filter:
                severity_list = ", ".join(sorted(severity_filter))
                lines.append(f"*Filtered by severity: {severity_list}*")
            lines.append("")

            # Group by severity
            for severity in ['HIGH', 'MEDIUM', 'LOW']:
                severity_issues = [i for i in filtered_issues if i.severity == severity]
                if not severity_issues:
                    continue

                emoji = self._get_severity_emoji(severity)
                lines.append(f"#### {emoji} {severity} Severity Issues")
                lines.append("")

                for issue in severity_issues:
                    lines.append(f"**{issue.test_name}** (`{issue.test_id}`)")
                    lines.append(f"- 📁 **File**: `{issue.filename}:{issue.line_number}`")
                    lines.append(f"- 🔍 **Confidence**: {issue.confidence}")
                    lines.append(f"- 📝 **Description**: {issue.issue_text}")

                    if include_code and issue.code:
                        lines.append(f"- 💻 **Code**:")
                        lines.append("```python")
                        lines.append(issue.code)
                        lines.append("```")

                    if issue.more_info:
                        lines.append(f"- 🔗 **More info**: {issue.more_info}")

                    lines.append("")
        else:
            lines.append(f"### 📋 Issue Summary")
            lines.append(f"*{len(filtered_issues)} issues found (showing summary due to large number)*")
            lines.append("")

            # Group by test type for summary
            test_counts = {}
            for issue in filtered_issues:
                test_name = issue.test_name
                if test_name not in test_counts:
                    test_counts[test_name] = {'count': 0, 'severity': issue.severity}
                test_counts[test_name]['count'] += 1

                # Keep highest severity
                current_level = self.SEVERITY_LEVELS.get(test_counts[test_name]['severity'], 0)
                issue_level = self.SEVERITY_LEVELS.get(issue.severity, 0)
                if issue_level > current_level:
                    test_counts[test_name]['severity'] = issue.severity

            for test_name, info in sorted(test_counts.items(),
                                          key=lambda x: self.SEVERITY_LEVELS.get(x[1]['severity'], 0),
                                          reverse=True):
                emoji = self._get_severity_emoji(info['severity'])
                lines.append(f"- {emoji} **{test_name}**: {info['count']} occurrences ({info['severity']})")

        # Recommendations
        lines.append("")
        lines.append("### 💡 Recommendations")
        lines.append("")

        if severity_summary['HIGH'] > 0:
            lines.append("🔴 **High priority**: Address HIGH severity issues immediately before deployment")

        if any(issue.test_id in self.OPS0_CRITICAL_PATTERNS for issue in self.issues):
            lines.append(
                "🧠 **MLOps specific**: Pay special attention to pickle usage and subprocess calls in ML pipelines")

        if severity_summary['MEDIUM'] > 0:
            lines.append("🟡 **Medium priority**: Review MEDIUM severity issues for production deployments")

        lines.append(
            "📚 **Documentation**: Check the [ops0 security guide](https://docs.ops0.xyz/security) for best practices")

        # Scan metadata
        lines.append("")
        lines.append("---")
        lines.append(f"*Scan completed at {datetime.now().isoformat()}*")

        if self.metrics:
            lines.append(f"*Files scanned: {self.metrics.get('_totals', {}).get('loc', 'Unknown')} lines of code*")

        return "\n".join(lines)

    def generate_json_summary(self) -> Dict:
        """Generate a JSON summary for programmatic use."""
        severity_summary = self.get_severity_summary()
        critical_issues = self.get_critical_ops0_issues()

        return {
            "scan_date": datetime.now().isoformat(),
            "total_issues": len(self.issues),
            "severity_breakdown": severity_summary,
            "critical_ops0_issues": len(critical_issues),
            "has_high_severity": severity_summary['HIGH'] > 0,
            "has_critical_patterns": len(critical_issues) > 0,
            "scan_passed": severity_summary['HIGH'] == 0 and len(critical_issues) == 0,
            "metrics": self.metrics,
            "issues": [
                {
                    "test_id": issue.test_id,
                    "test_name": issue.test_name,
                    "filename": issue.filename,
                    "line_number": issue.line_number,
                    "severity": issue.severity,
                    "confidence": issue.confidence,
                    "issue_text": issue.issue_text,
                    "is_critical_for_ops0": issue.test_id in self.OPS0_CRITICAL_PATTERNS
                }
                for issue in self.issues
            ]
        }

    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for severity level."""
        emoji_map = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢',
            'UNDEFINED': '⚪'
        }
        return emoji_map.get(severity.upper(), '❓')


def main():
    """Main function to process Bandit reports."""
    parser = argparse.ArgumentParser(
        description="Process Bandit security scan reports for ops0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_bandit_report.py bandit-report.json
  python process_bandit_report.py bandit-report.json --format json
  python process_bandit_report.py bandit-report.json --severity HIGH,MEDIUM
  python process_bandit_report.py bandit-report.json --output security-summary.md
        """
    )

    parser.add_argument(
        'report_file',
        help='Path to the Bandit JSON report file'
    )

    parser.add_argument(
        '--format',
        choices=['markdown', 'json'],
        default='markdown',
        help='Output format (default: markdown)'
    )

    parser.add_argument(
        '--severity',
        help='Filter by severity levels (comma-separated: HIGH,MEDIUM,LOW)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: stdout)'
    )

    parser.add_argument(
        '--include-code',
        action='store_true',
        default=True,
        help='Include code snippets in markdown output'
    )

    parser.add_argument(
        '--exclude-files',
        help='Exclude files matching patterns (comma-separated)'
    )

    args = parser.parse_args()

    # Initialize processor
    processor = BanditReportProcessor(args.report_file)

    # Load report
    if not processor.load_report():
        sys.exit(1)

    # Parse severity filter
    severity_filter = None
    if args.severity:
        severity_filter = set(s.strip().upper() for s in args.severity.split(','))
        valid_severities = {'HIGH', 'MEDIUM', 'LOW'}
        invalid_severities = severity_filter - valid_severities
        if invalid_severities:
            print(f"Warning: Invalid severity levels: {invalid_severities}")
            severity_filter = severity_filter & valid_severities

    # Parse exclude filter
    exclude_files = None
    if args.exclude_files:
        exclude_files = set(s.strip() for s in args.exclude_files.split(','))

    # Generate output
    if args.format == 'json':
        output = json.dumps(processor.generate_json_summary(), indent=2)
    else:
        output = processor.generate_markdown_summary(
            severity_filter=severity_filter,
            include_code=args.include_code
        )

    # Write output
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Report written to {args.output}")
        except Exception as e:
            print(f"Error writing output file: {e}")
            sys.exit(1)
    else:
        print(output)

    # Exit with error code if high severity issues found
    if args.format == 'json':
        summary = processor.generate_json_summary()
        if not summary['scan_passed']:
            sys.exit(1)
    else:
        severity_summary = processor.get_severity_summary()
        if severity_summary['HIGH'] > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
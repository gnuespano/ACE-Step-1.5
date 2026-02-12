"""
Side-Step TUI Widgets

Reusable UI components for the TUI application:
- GPUGauge: Live GPU/VRAM utilization gauge
- LossSparkline: Mini loss graph widget
- ConfigForm: Reusable configuration input forms
- FilePicker: Directory/file browser widget
- LogViewer: Scrolling log panel
- StatusBar: Bottom status bar
"""

from __future__ import annotations

__all__ = [
    "GPUGauge",
    "LossSparkline",
    "FilePicker",
    "LogViewer",
]

# Lazy imports
def __getattr__(name: str):
    """
    Lazily load and return a requested TUI widget class from this package's submodules.
    
    Parameters:
        name (str): The attribute name to load. Supported values are "GPUGauge", "LossSparkline", "FilePicker", and "LogViewer".
    
    Returns:
        type: The widget class object corresponding to `name`.
    
    Raises:
        AttributeError: If `name` is not one of the supported widget names.
    """
    if name == "GPUGauge":
        from acestep.training_v2.tui.widgets.gpu_gauge import GPUGauge
        return GPUGauge
    elif name == "LossSparkline":
        from acestep.training_v2.tui.widgets.loss_sparkline import LossSparkline
        return LossSparkline
    elif name == "FilePicker":
        from acestep.training_v2.tui.widgets.file_picker import FilePicker
        return FilePicker
    elif name == "LogViewer":
        from acestep.training_v2.tui.widgets.log_viewer import LogViewer
        return LogViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
"""
GPU Gauge Widget

Live GPU utilization and VRAM usage gauge with:
- Animated progress bars
- Temperature and power display
- Auto-refresh capability
"""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, ProgressBar, Label
from textual.widget import Widget
from textual.reactive import reactive


class GPUGauge(Widget):
    """
    GPU status gauge showing VRAM usage, utilization, and other metrics.
    """
    
    DEFAULT_CSS = """
    GPUGauge {
        height: auto;
        padding: 0 1;
        border: round $primary 30%;
    }
    
    #gauge-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .gauge-row {
        height: 2;
        layout: horizontal;
        margin-bottom: 0;
    }
    
    .gauge-label {
        width: 12;
        color: $text-muted;
    }
    
    .gauge-bar {
        width: 1fr;
    }
    
    .gauge-value {
        width: 14;
        text-align: right;
    }
    
    #gauge-extras {
        height: auto;
        margin-top: 1;
        layout: horizontal;
    }
    
    .extra-stat {
        width: 1fr;
        text-align: center;
    }
    
    ProgressBar {
        width: 100%;
    }
    
    ProgressBar > .bar--complete {
        color: $success;
    }
    
    ProgressBar > .bar--indeterminate {
        color: $primary;
    }
    """
    
    # Reactive properties for auto-updating display
    vram_used: reactive[float] = reactive(0.0)
    vram_total: reactive[float] = reactive(0.0)
    utilization: reactive[float] = reactive(0.0)
    temperature: reactive[float] = reactive(0.0)
    power: reactive[float] = reactive(0.0)
    gpu_name: reactive[str] = reactive("GPU")
    
    def __init__(
        self,
        auto_refresh: bool = True,
        refresh_interval: float = 2.0,
        show_extras: bool = True,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        """
        Create a GPUGauge widget and configure automatic refresh behavior and optional extra stats.
        
        Parameters:
            auto_refresh (bool): If True, start periodic GPU stat refresh on mount.
            refresh_interval (float): Seconds between automatic refreshes.
            show_extras (bool): If True, display temperature and power stats.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        self.show_extras = show_extras
    
    def compose(self) -> ComposeResult:
        """
        Builds and yields the widget tree for the GPU gauge.
        
        Creates a title and two primary rows: a VRAM row (label, progress bar with id `vram-bar`, and value with id `vram-value`) and a Utilization row (label, progress bar with id `util-bar`, and value with id `util-value`). If `self.show_extras` is True, also creates an extras row containing temperature (`temp-stat`) and power (`power-stat`) statics.
        
        Returns:
            ComposeResult: A generator yielding the Textual widgets that form the gauge.
        """
        yield Static("GPU Status", id="gauge-title")
        
        # VRAM gauge
        with Horizontal(classes="gauge-row"):
            yield Static("VRAM", classes="gauge-label")
            with Container(classes="gauge-bar"):
                yield ProgressBar(total=100, show_eta=False, id="vram-bar")
            yield Static("0 / 0 GB", id="vram-value", classes="gauge-value")
        
        # Utilization gauge
        with Horizontal(classes="gauge-row"):
            yield Static("Utilization", classes="gauge-label")
            with Container(classes="gauge-bar"):
                yield ProgressBar(total=100, show_eta=False, id="util-bar")
            yield Static("0%", id="util-value", classes="gauge-value")
        
        # Extra stats (temperature, power)
        if self.show_extras:
            with Horizontal(id="gauge-extras"):
                yield Static("Temp: --°C", id="temp-stat", classes="extra-stat")
                yield Static("Power: --W", id="power-stat", classes="extra-stat")
    
    def on_mount(self) -> None:
        """
        Initialize the widget when mounted and enable periodic GPU stat refresh if auto-refresh is enabled.
        
        If the widget is configured for auto-refresh, perform an immediate GPU stats refresh and schedule recurring refreshes at the configured refresh_interval.
        """
        if self.auto_refresh:
            self._refresh_gpu_stats()
            self.set_interval(self.refresh_interval, self._refresh_gpu_stats)
    
    def _refresh_gpu_stats(self) -> None:
        """
        Fetch current GPU metrics and update the widget's reactive GPU fields.
        
        Updates vram_used, vram_total, utilization, temperature, power, and gpu_name from the latest GPU info. If fetching fails or an exception occurs, previous values are preserved and no exception is propagated.
        """
        try:
            from acestep.training_v2.gpu_utils import get_gpu_info
            info = get_gpu_info()
            
            self.vram_used = info.get("vram_used_gb", 0)
            self.vram_total = info.get("vram_total_gb", 0)
            self.utilization = info.get("utilization", 0)
            self.temperature = info.get("temperature", 0)
            self.power = info.get("power", 0)
            self.gpu_name = info.get("name", "GPU")
        except Exception:
            pass  # Keep previous values on error
    
    def watch_vram_used(self, value: float) -> None:
        """
        Update the VRAM display when the tracked VRAM used value changes.
        
        Parameters:
            value (float): New VRAM used value in gigabytes.
        """
        self._update_vram_display()
    
    def watch_vram_total(self, value: float) -> None:
        """
        Handle updates to the total VRAM value and refresh the VRAM display.
        
        Parameters:
            value (float): New total VRAM in gigabytes.
        """
        self._update_vram_display()
    
    def watch_utilization(self, value: float) -> None:
        """
        Update the utilization progress bar and its percentage label when utilization changes.
        
        Parameters:
            value (float): GPU utilization percentage (0–100) to display.
        """
        try:
            bar = self.query_one("#util-bar", ProgressBar)
            bar.progress = value
            self.query_one("#util-value", Static).update(f"{value:.0f}%")
        except Exception:
            pass
    
    def watch_temperature(self, value: float) -> None:
        """
        Update the visible temperature text when the temperature value changes.
        
        Parameters:
            value (float): Temperature in degrees Celsius to display (rounded to nearest integer).
        """
        if self.show_extras:
            try:
                self.query_one("#temp-stat", Static).update(f"Temp: {value:.0f}°C")
            except Exception:
                pass
    
    def watch_power(self, value: float) -> None:
        """
        Update the displayed GPU power statistic in the extras area.
        
        Parameters:
            value (float): Power in watts; displayed rounded to the nearest whole watt (e.g., `42W`).
        """
        if self.show_extras:
            try:
                self.query_one("#power-stat", Static).update(f"Power: {value:.0f}W")
            except Exception:
                pass
    
    def _update_vram_display(self) -> None:
        """
        Update the displayed VRAM progress bar and numeric VRAM value.
        
        Sets the VRAM progress bar's progress to the percentage of VRAM used and updates the VRAM text to the format "<used:.1f>/<total:.0f>GB".
        """
        try:
            if self.vram_total > 0:
                percent = (self.vram_used / self.vram_total) * 100
            else:
                percent = 0
            
            bar = self.query_one("#vram-bar", ProgressBar)
            bar.progress = percent
            
            self.query_one("#vram-value", Static).update(
                f"{self.vram_used:.1f}/{self.vram_total:.0f}GB"
            )
        except Exception:
            pass
    
    def set_stats(
        self,
        vram_used: float,
        vram_total: float,
        utilization: float = 0,
        temperature: float = 0,
        power: float = 0,
    ) -> None:
        """
        Set the GPU statistics displayed by the widget.
        
        Parameters:
            vram_used (float): VRAM used in gigabytes.
            vram_total (float): Total VRAM in gigabytes.
            utilization (float): GPU utilization as a percentage.
            temperature (float): GPU temperature in degrees Celsius.
            power (float): Power draw in watts.
        """
        self.vram_used = vram_used
        self.vram_total = vram_total
        self.utilization = utilization
        self.temperature = temperature
        self.power = power


class MiniGPUGauge(Widget):
    """
    Compact GPU gauge for headers/status bars.
    Shows just VRAM and utilization in a single line.
    """
    
    DEFAULT_CSS = """
    MiniGPUGauge {
        height: 1;
        width: auto;
        layout: horizontal;
    }
    
    MiniGPUGauge Static {
        margin-right: 2;
    }
    """
    
    vram_percent: reactive[float] = reactive(0.0)
    utilization: reactive[float] = reactive(0.0)
    
    def __init__(
        self,
        auto_refresh: bool = True,
        refresh_interval: float = 2.0,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        """
        Initialize the MiniGPUGauge widget and configure its automatic refresh behavior.
        
        Parameters:
            auto_refresh (bool): If True, the widget will periodically fetch GPU stats on mount and schedule recurring updates.
            refresh_interval (float): Number of seconds between automatic refreshes when `auto_refresh` is enabled.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
    
    def compose(self) -> ComposeResult:
        """
        Create the widget tree for the compact GPU gauge.
        
        Yields two Static widgets arranged inline: a utilization label with id `"mini-util"` initialized to `"GPU: --%"` and a VRAM label with id `"mini-vram"` initialized to `"VRAM: --GB"`.
        
        Returns:
            ComposeResult: an iterator yielding the two Static widgets for the MiniGPUGauge.
        """
        yield Static("GPU: --%", id="mini-util")
        yield Static("VRAM: --GB", id="mini-vram")
    
    def on_mount(self) -> None:
        """
        Start automatic GPU stat refresh when the widget is mounted.
        
        If `auto_refresh` is True, perform an immediate stats update and schedule recurring updates at `refresh_interval` seconds.
        """
        if self.auto_refresh:
            self._refresh()
            self.set_interval(self.refresh_interval, self._refresh)
    
    def _refresh(self) -> None:
        """
        Refreshes GPU utilization and VRAM readings and updates the mini gauge display.
        
        Queries GPU info, updates the widget's `utilization` and `vram_percent` reactive values, and writes formatted text to the Static elements "#mini-util" ("GPU: <util>%") and "#mini-vram" ("VRAM: <used>GB"). Any exceptions raised during the refresh are caught and ignored to preserve the existing display.
        """
        try:
            from acestep.training_v2.gpu_utils import get_gpu_info
            info = get_gpu_info()
            
            util = info.get("utilization", 0)
            vram_used = info.get("vram_used_gb", 0)
            vram_total = info.get("vram_total_gb", 0)
            
            if vram_total > 0:
                self.vram_percent = (vram_used / vram_total) * 100
            
            self.utilization = util
            
            self.query_one("#mini-util", Static).update(f"GPU: {util:.0f}%")
            self.query_one("#mini-vram", Static).update(
                f"VRAM: {vram_used:.1f}GB"
            )
        except Exception:
            pass
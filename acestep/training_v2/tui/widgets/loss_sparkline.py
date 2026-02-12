"""
Loss Sparkline Widget

Mini loss graph showing training progress with:
- ASCII-based sparkline visualization
- Rolling window of recent values
- Best/current/average annotations
"""

from __future__ import annotations

from typing import Optional, List
from collections import deque

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Sparkline, Label
from textual.widget import Widget
from textual.reactive import reactive


# Sparkline characters for different resolutions
SPARK_CHARS_8 = "▁▂▃▄▅▆▇█"
SPARK_CHARS_4 = "▁▃▅█"


class LossSparkline(Widget):
    """
    Mini loss graph widget showing training loss over time.
    
    Uses ASCII sparkline characters for a compact visualization.
    """
    
    DEFAULT_CSS = """
    LossSparkline {
        height: auto;
        min-height: 6;
        border: round $primary 30%;
        padding: 1;
    }
    
    #sparkline-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    #sparkline-graph {
        height: 3;
        width: 100%;
    }
    
    #sparkline-axis {
        height: 1;
        color: $text-muted;
    }
    
    #sparkline-stats {
        height: auto;
        margin-top: 1;
        layout: horizontal;
    }
    
    .stat-item {
        width: 1fr;
        text-align: center;
    }
    
    .stat-label {
        color: $text-muted;
    }
    
    .stat-value {
        text-style: bold;
    }
    
    .current-loss {
        color: $primary;
    }
    
    .best-loss {
        color: $success;
    }
    """
    
    # Reactive properties
    current_loss: reactive[float] = reactive(0.0)
    best_loss: reactive[float] = reactive(float("inf"))
    best_epoch: reactive[int] = reactive(0)
    
    def __init__(
        self,
        max_points: int = 100,
        title: str = "Loss",
        show_stats: bool = True,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        """
        Create a LossSparkline widget configured with a rolling data window and display options.
        
        Parameters:
            max_points (int): Maximum number of loss values retained for the sparkline.
            title (str): Header text shown above the sparkline.
            show_stats (bool): If True, display current, best, and average statistics.
            name (Optional[str]): Optional widget name.
            id (Optional[str]): Optional widget identifier.
            classes (Optional[str]): Optional CSS classes applied to the widget.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.max_points = max_points
        self.title = title
        self.show_stats = show_stats
        self._data: deque[float] = deque(maxlen=max_points)
    
    def compose(self) -> ComposeResult:
        """
        Builds the widget's child components: title, graph area, axis, and an optional statistics block.
        
        When `show_stats` is True the statistics block contains three labeled value placeholders for
        Current, Best, and Average.
        
        Returns:
            ComposeResult: an iterator that yields the child widgets for the widget layout.
        """
        yield Static(self.title, id="sparkline-title")
        yield Static("", id="sparkline-graph")
        yield Static("", id="sparkline-axis")
        
        if self.show_stats:
            with Horizontal(id="sparkline-stats"):
                with Vertical(classes="stat-item"):
                    yield Static("Current", classes="stat-label")
                    yield Static("--", id="stat-current", classes="stat-value current-loss")
                with Vertical(classes="stat-item"):
                    yield Static("Best", classes="stat-label")
                    yield Static("--", id="stat-best", classes="stat-value best-loss")
                with Vertical(classes="stat-item"):
                    yield Static("Average", classes="stat-label")
                    yield Static("--", id="stat-avg", classes="stat-value")
    
    def add_value(self, value: float, epoch: Optional[int] = None) -> None:
        """
        Append a loss value to the rolling window and update current/best metrics.
        
        Updates the widget's stored values with the new loss, updates the recorded best loss
        (and associated epoch if provided), and refreshes the widget display.
        
        Parameters:
            value (float): Loss value to append.
            epoch (Optional[int]): Epoch number to associate with the value when updating the best loss.
        """
        self._data.append(value)
        self.current_loss = value
        
        if value < self.best_loss:
            self.best_loss = value
            if epoch is not None:
                self.best_epoch = epoch
        
        self._update_display()
    
    def add_values(self, values: List[float]) -> None:
        """
        Append multiple loss values to the widget's rolling buffer and refresh the display.
        
        Parameters:
        	values (List[float]): Sequence of loss values to append. If non-empty, the widget's `current_loss` is set to the last value and `best_loss` is updated when a lower value is present. The visual sparkline, axis, and stats are updated after values are added.
        """
        for v in values:
            self._data.append(v)
        
        if values:
            self.current_loss = values[-1]
            min_val = min(values)
            if min_val < self.best_loss:
                self.best_loss = min_val
        
        self._update_display()
    
    def clear(self) -> None:
        """
        Reset stored loss data and associated statistics and refresh the widget display.
        
        Clears the internal data buffer, sets current_loss to 0.0, best_loss to infinity, and best_epoch to 0, then updates the visual representation.
        """
        self._data.clear()
        self.current_loss = 0.0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self._update_display()
    
    def _update_display(self) -> None:
        """
        Refresh the widget's displayed sparkline, axis, and statistics from the current stored data.
        
        If no data is available, clear the graph display. Otherwise regenerate the sparkline and axis visuals and, when `show_stats` is true, update the current/best/average statistics.
        """
        if not self._data:
            self.query_one("#sparkline-graph", Static).update("")
            return
        
        # Generate sparkline
        sparkline = self._generate_sparkline()
        self.query_one("#sparkline-graph", Static).update(sparkline)
        
        # Generate axis
        axis = self._generate_axis()
        self.query_one("#sparkline-axis", Static).update(axis)
        
        # Update stats
        if self.show_stats:
            self._update_stats()
    
    def _generate_sparkline(self) -> str:
        """
        Generate an ASCII sparkline representing the widget's stored data.
        
        The returned string is built from the widget's internal data window. If the stored data length exceeds the available graph width the data is downsampled to fit; values are normalized and mapped to characters from SPARK_CHARS_8. If all sampled values are equal, a repeated middle-high spark character is returned. If there is no data, an empty string is returned.
        
        Returns:
            sparkline (str): ASCII sparkline for the current data or an empty string when no data is available.
        """
        if not self._data:
            return ""
        
        data = list(self._data)
        
        # Get terminal width (approximate)
        try:
            graph = self.query_one("#sparkline-graph", Static)
            width = graph.size.width - 2  # Account for padding
        except Exception:
            width = 60
        
        # Sample data to fit width
        if len(data) > width:
            step = len(data) / width
            sampled = [data[int(i * step)] for i in range(width)]
        else:
            sampled = data
        
        if not sampled:
            return ""
        
        # Normalize to sparkline range
        min_val = min(sampled)
        max_val = max(sampled)
        
        if max_val == min_val:
            # All values the same
            return SPARK_CHARS_8[4] * len(sampled)
        
        # Map to character indices
        chars = SPARK_CHARS_8
        range_val = max_val - min_val
        
        sparkline = ""
        for v in sampled:
            normalized = (v - min_val) / range_val
            idx = int(normalized * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            sparkline += chars[idx]
        
        return sparkline
    
    def _generate_axis(self) -> str:
        """
        Create an axis string showing the data minimum and maximum aligned to the sparkline width.
        
        If there is no stored data an empty string is returned. The minimum and maximum values are formatted with three decimal places and separated by spaces so the combined length matches the sparkline display width (uses the graph widget width minus two; falls back to 60 characters if the widget size cannot be determined).
        
        Returns:
            A string containing the formatted minimum and maximum values separated by padding spaces to fit the sparkline width, or an empty string if no data is present.
        """
        if not self._data:
            return ""
        
        min_val = min(self._data)
        max_val = max(self._data)
        
        try:
            graph = self.query_one("#sparkline-graph", Static)
            width = graph.size.width - 2
        except Exception:
            width = 60
        
        # Create axis with min/max labels
        min_str = f"{min_val:.3f}"
        max_str = f"{max_val:.3f}"
        
        # Calculate spacing
        padding = width - len(min_str) - len(max_str)
        if padding < 0:
            padding = 0
        
        return f"{min_str}{' ' * padding}{max_str}"
    
    def _update_stats(self) -> None:
        """
        Refresh the widget's current, best, and average loss displays in the UI.
        
        Updates the "#stat-current" label with the current loss formatted to four decimals, the "#stat-best" label with the best loss formatted to four decimals (appending " @ <epoch>" when a best epoch is recorded) or "--" if no best exists, and the "#stat-avg" label with the average of stored values formatted to four decimals or "--" if no data is available. Any errors raised while updating the UI are swallowed.
        """
        try:
            # Current
            current_text = f"{self.current_loss:.4f}"
            self.query_one("#stat-current", Static).update(current_text)
            
            # Best
            if self.best_loss < float("inf"):
                best_text = f"{self.best_loss:.4f}"
                if self.best_epoch > 0:
                    best_text += f" @ {self.best_epoch}"
            else:
                best_text = "--"
            self.query_one("#stat-best", Static).update(best_text)
            
            # Average
            if self._data:
                avg = sum(self._data) / len(self._data)
                avg_text = f"{avg:.4f}"
            else:
                avg_text = "--"
            self.query_one("#stat-avg", Static).update(avg_text)
        except Exception:
            pass


class MiniSparkline(Widget):
    """
    Very compact sparkline for embedding in tables or lists.
    Just shows the graph, no labels.
    """
    
    DEFAULT_CSS = """
    MiniSparkline {
        height: 1;
        width: auto;
        min-width: 20;
    }
    """
    
    def __init__(
        self,
        max_points: int = 20,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        """
        Initialize the mini sparkline widget and its rolling data buffer.
        
        Parameters:
            max_points (int): Maximum number of values to retain in the rolling window.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.max_points = max_points
        self._data: deque[float] = deque(maxlen=max_points)
    
    def compose(self) -> ComposeResult:
        """
        Builds the widget tree for the mini sparkline.
        
        Yields a single Static placeholder (id="mini-spark") that hosts the compact sparkline rendering.
        
        Returns:
            ComposeResult: An iterator that yields the Static widget for the mini sparkline.
        """
        yield Static("", id="mini-spark")
    
    def add_value(self, value: float) -> None:
        """
        Append a numeric sample to the sparkline's rolling data window.
        
        Parameters:
            value (float): The measurement to add (e.g., a loss value); the newest value becomes the current sample.
        """
        self._data.append(value)
        self._update()
    
    def _update(self) -> None:
        """
        Render the widget's stored values as a compact 4-level ASCII sparkline and write it to the #mini-spark element.
        
        Converts the current rolling data into a sparkline using the four-character palette, uses the middle palette character when all values are equal, updates the #mini-spark Static element, and silently ignores any errors raised while updating the UI.
        """
        if not self._data:
            return
        
        data = list(self._data)
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            sparkline = SPARK_CHARS_4[2] * len(data)
        else:
            chars = SPARK_CHARS_4
            range_val = max_val - min_val
            sparkline = ""
            for v in data:
                normalized = (v - min_val) / range_val
                idx = int(normalized * (len(chars) - 1))
                idx = max(0, min(len(chars) - 1, idx))
                sparkline += chars[idx]
        
        try:
            self.query_one("#mini-spark", Static).update(sparkline)
        except Exception:
            pass
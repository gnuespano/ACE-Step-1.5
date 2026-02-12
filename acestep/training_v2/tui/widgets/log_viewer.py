"""
Log Viewer Widget

Scrolling log panel for training output with:
- Auto-scroll to bottom
- Timestamp formatting
- Log level coloring
- Search capability
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static, RichLog
from textual.widget import Widget
from textual.binding import Binding

from rich.text import Text


class LogViewer(Widget):
    """
    Scrolling log viewer widget for training output.
    """
    
    DEFAULT_CSS = """
    LogViewer {
        height: 100%;
        border: round $primary 30%;
    }
    
    #log-header {
        dock: top;
        height: 1;
        background: $panel;
        padding: 0 1;
    }
    
    #log-content {
        height: 1fr;
    }
    
    RichLog {
        height: 100%;
    }
    """
    
    BINDINGS = [
        Binding("g", "scroll_home", "Top"),
        Binding("G", "scroll_end", "Bottom"),
        Binding("c", "clear", "Clear"),
    ]
    
    def __init__(
        self,
        title: str = "Log",
        max_lines: int = 1000,
        auto_scroll: bool = True,
        show_timestamps: bool = True,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        """
        Create a LogViewer widget configured with display and buffer behavior.
        
        Parameters:
            title (str): Header text shown above the log area.
            max_lines (int): Maximum number of log lines retained in the internal buffer.
            auto_scroll (bool): If true, the view scrolls to the bottom when new entries are added.
            show_timestamps (bool): If true, prepend a HH:MM:SS timestamp to each log entry.
            name (Optional[str]): Optional widget name.
            id (Optional[str]): Optional widget identifier.
            classes (Optional[str]): Optional CSS classes applied to the widget.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.title = title
        self.max_lines = max_lines
        self.auto_scroll = auto_scroll
        self.show_timestamps = show_timestamps
        self._line_count = 0
    
    def compose(self) -> ComposeResult:
        """Compose the log viewer layout."""
        yield Static(self.title, id="log-header")
        with Container(id="log-content"):
            yield RichLog(
                highlight=True,
                markup=True,
                max_lines=self.max_lines,
                id="rich-log",
            )
    
    def write(
        self,
        message: str,
        level: str = "info",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Write a message to the log.
        
        Args:
            message: The message to write
            level: Log level (info, warning, error, debug, success)
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format with timestamp
        if self.show_timestamps:
            time_str = timestamp.strftime("%H:%M:%S")
            prefix = f"[dim][{time_str}][/dim] "
        else:
            prefix = ""
        
        # Apply level coloring
        level_styles = {
            "info": "",
            "warning": "[yellow]",
            "error": "[red bold]",
            "debug": "[dim]",
            "success": "[green]",
        }
        
        style = level_styles.get(level, "")
        end_style = "[/]" if style else ""
        
        formatted = f"{prefix}{style}{message}{end_style}"
        
        try:
            log = self.query_one("#rich-log", RichLog)
            log.write(formatted)
            self._line_count += 1
        except Exception:
            pass
    
    def write_line(self, message: str) -> None:
        """
        Write a log entry using the viewer's default log level.
        
        Parameters:
            message (str): Text to append as a single log line.
        """
        self.write(message)
    
    def info(self, message: str) -> None:
        """
        Log a message with informational severity.
        
        Parameters:
            message (str): Text to write to the log.
        """
        self.write(message, level="info")
    
    def warning(self, message: str) -> None:
        """Log a message at the warning level."""
        self.write(message, level="warning")
    
    def error(self, message: str) -> None:
        """
        Log a message at the error level.
        """
        self.write(message, level="error")
    
    def debug(self, message: str) -> None:
        """Write a debug message."""
        self.write(message, level="debug")
    
    def success(self, message: str) -> None:
        """
        Log a message styled as a successful notification.
        
        Parameters:
            message (str): Text to log as a success message.
        """
        self.write(message, level="success")
    
    def write_separator(self, char: str = "─") -> None:
        """
        Write a dim separator line into the log using the given character.
        
        Parameters:
            char (str): Character to repeat for the separator (default "─"). The separator length is based on the current RichLog width with a fallback of 40.
        """
        try:
            log = self.query_one("#rich-log", RichLog)
            # Get approximate width
            width = log.size.width - 4
            if width < 10:
                width = 40
            log.write(f"[dim]{char * width}[/dim]")
        except Exception:
            pass
    
    def write_header(self, title: str) -> None:
        """
        Write a visually separated bold section header into the log.
        
        Parameters:
            title (str): Header text to display.
        """
        self.write_separator()
        self.write(f"[bold]{title}[/bold]", level="info")
        self.write_separator()
    
    def clear(self) -> None:
        """
        Clear the viewer's log entries and reset the internal line counter.
        
        If the RichLog widget is not available or an error occurs while clearing, this method returns silently without raising.
        """
        try:
            log = self.query_one("#rich-log", RichLog)
            log.clear()
            self._line_count = 0
        except Exception:
            pass
    
    def action_clear(self) -> None:
        """Action to clear the log."""
        self.clear()
    
    def action_scroll_home(self) -> None:
        """Scroll to top of log."""
        try:
            log = self.query_one("#rich-log", RichLog)
            log.scroll_home()
        except Exception:
            pass
    
    def action_scroll_end(self) -> None:
        """Scroll to bottom of log."""
        try:
            log = self.query_one("#rich-log", RichLog)
            log.scroll_end()
        except Exception:
            pass


class TrainingLogViewer(LogViewer):
    """
    Specialized log viewer for training output.
    
    Adds training-specific formatting and parsing.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initialize a TrainingLogViewer widget with training-focused defaults.
        
        Parameters:
            title (str, optional): Header title for the viewer; defaults to "Training Log".
            **kwargs: All other keyword arguments are forwarded to the base LogViewer constructor.
        
        Behavior:
            Sets internal epoch tracking (_current_epoch) to 0.
        """
        kwargs.setdefault("title", "Training Log")
        super().__init__(**kwargs)
        self._current_epoch = 0
    
    def log_epoch_start(self, epoch: int, total: int) -> None:
        """
        Record and display the start of a training epoch in the log.
        
        Parameters:
            epoch (int): Current epoch number.
            total (int): Total number of epochs.
        """
        self._current_epoch = epoch
        self.write_separator("═")
        self.write(f"Epoch {epoch}/{total} started", level="info")
    
    def log_epoch_end(self, epoch: int, loss: float, lr: float) -> None:
        """
        Log that an epoch finished and emit a success message including epoch number, loss, and learning rate.
        
        Parameters:
            epoch (int): Completed epoch number.
            loss (float): Final loss for the epoch.
            lr (float): Learning rate used during the epoch.
        """
        self.success(f"Epoch {epoch} completed | Loss: {loss:.4f} | LR: {lr:.2e}")
    
    def log_step(self, step: int, total_steps: int, loss: float) -> None:
        """
        Record a single training step with its loss.
        
        Parameters:
        	step (int): Current step index (1-based or epoch-local step number).
        	total_steps (int): Total number of steps in the current epoch or loop.
        	loss (float): Loss value for this step, formatted to four decimal places when logged.
        """
        self.debug(f"Step {step}/{total_steps} | Loss: {loss:.4f}")
    
    def log_checkpoint_saved(self, path: str) -> None:
        """
        Log a message indicating a training checkpoint was saved.
        
        Parameters:
            path (str): Path to the saved checkpoint file.
        """
        self.success(f"Checkpoint saved: {path}")
    
    def log_best_model(self, epoch: int, loss: float) -> None:
        """
        Record a new best-model message for an epoch.
        
        Parameters:
            epoch (int): Epoch number when the new best model was found.
            loss (float): Loss value for the best model; displayed with four decimal places.
        """
        self.success(f"★ New best model @ epoch {epoch} | Loss: {loss:.4f}")
    
    def log_training_complete(self, epochs: int, best_loss: float, time_str: str) -> None:
        """
        Emit a final training summary to the log including totals and elapsed time.
        
        Parameters:
            epochs (int): Total number of training epochs completed.
            best_loss (float): Best observed loss value.
            time_str (str): Human-readable total training duration.
        """
        self.write_separator("═")
        self.success(f"Training complete!")
        self.info(f"Total epochs: {epochs}")
        self.info(f"Best loss: {best_loss:.4f}")
        self.info(f"Total time: {time_str}")
        self.write_separator("═")
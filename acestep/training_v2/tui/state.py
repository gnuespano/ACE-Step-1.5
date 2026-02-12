"""
Side-Step TUI State Management

Centralized reactive state for the application, including:
- Current training run information
- Recent runs history
- User preferences
- GPU status
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime


@dataclass
class RunInfo:
    """Information about a training run."""
    name: str
    trainer_type: str  # "fixed", "vanilla", "selective"
    status: str  # "running", "completed", "failed", "paused"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    current_epoch: int = 0
    total_epochs: int = 100
    current_loss: float = 0.0
    best_loss: float = float("inf")
    best_epoch: int = 0
    output_dir: str = ""
    checkpoint_dir: str = ""
    dataset_dir: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of this dataclass.
        
        Returns:
            dict: Mapping of field names to their values.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunInfo":
        """
        Create a RunInfo from a dictionary of attributes.
        
        Parameters:
            data (Dict[str, Any]): Mapping with keys matching RunInfo constructor parameters.
        
        Returns:
            RunInfo: An instance populated from the provided dictionary.
        """
        return cls(**data)


@dataclass
class GPUStatus:
    """Current GPU status information."""
    name: str = "Unknown"
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    utilization_percent: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    
    @property
    def vram_percent(self) -> float:
        """
        Compute VRAM usage as a percentage of total VRAM.
        
        Returns:
            float: Percentage of VRAM used (0.0 if total VRAM is zero), in the range 0.0–100.0.
        """
        if self.vram_total_gb == 0:
            return 0.0
        return (self.vram_used_gb / self.vram_total_gb) * 100


@dataclass
class UserPreferences:
    """User preferences and settings."""
    default_checkpoint_dir: str = "./checkpoints"
    default_output_dir: str = "./lora_output"
    default_dataset_dir: str = "./datasets"
    theme: str = "dark"
    show_gpu_in_header: bool = True
    auto_save_config: bool = True
    confirm_on_quit: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of this dataclass.
        
        Returns:
            dict: Mapping of field names to their values.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """
        Create a UserPreferences instance from a mapping of persisted preference values.
        
        Parameters:
            data (Dict[str, Any]): Dictionary of preference keys and values; keys not defined on UserPreferences are ignored.
        
        Returns:
            UserPreferences: An instance populated with the values from `data`.
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AppState:
    """
    Centralized application state with reactive updates.
    
    Manages:
    - Current training run
    - Recent runs history
    - GPU status
    - User preferences
    """
    
    @staticmethod
    def _resolve_config_dir() -> Path:
        """
        Determine the platform-appropriate configuration directory for the application.
        
        On Windows this uses the APPDATA environment variable when available; on other platforms it uses the user's ~/.config directory.
        
        Returns:
            Path: Path to the application's configuration directory (a subdirectory named "sidestep").
        """
        import os
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:
            base = Path.home() / ".config"
        return base / "sidestep"
    
    def __init__(self) -> None:
        """
        Initialize the AppState, set up file paths and in-memory state, and load persisted configuration and run history.
        
        This constructs platform-aware paths for configuration and history files, initializes runtime fields for the current run, recent runs, GPU status, user preferences, event listeners, and the last estimation path, then loads saved preferences and recent run history from disk.
        """
        self.CONFIG_DIR = self._resolve_config_dir()
        self.CONFIG_FILE = self.CONFIG_DIR / "config.json"
        self.HISTORY_FILE = self.CONFIG_DIR / "history.json"
        
        self._current_run: Optional[RunInfo] = None
        self._recent_runs: List[RunInfo] = []
        self._gpu_status = GPUStatus()
        self._preferences = UserPreferences()
        self._listeners: Dict[str, List[Callable]] = {}
        self._last_estimation: Optional[str] = None  # Path to last estimation JSON
        
        # Load saved state
        self._load_config()
        self._load_history()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def current_run(self) -> Optional[RunInfo]:
        """
        Get the active training run being tracked by the application.
        
        Returns:
            The current RunInfo instance if a run is active or paused, `None` if there is no current run.
        """
        return self._current_run
    
    @current_run.setter
    def current_run(self, value: Optional[RunInfo]) -> None:
        """
        Set the current RunInfo for the application and notify listeners of the change.
        
        Parameters:
            value (Optional[RunInfo]): The RunInfo to set as current; pass `None` to clear the current run.
        """
        self._current_run = value
        self._notify("current_run", value)
    
    @property
    def recent_runs(self) -> List[RunInfo]:
        """
        Return the list of recent RunInfo records maintained by the application.
        
        Returns:
            List[RunInfo]: Recent runs, ordered with the most-recent (newest) run first.
        """
        return self._recent_runs
    
    @property
    def gpu_status(self) -> GPUStatus:
        """
        Current GPU status snapshot.
        
        Returns:
            GPUStatus: An object containing current GPU metrics (name, VRAM used and total in GB, VRAM utilization percent, temperature in °C, and power in W).
        """
        return self._gpu_status
    
    @gpu_status.setter
    def gpu_status(self, value: GPUStatus) -> None:
        """
        Set the current GPU status and notify subscribers of the update.
        
        Updates the stored GPUStatus to `value` and emits a "gpu_status" event to registered listeners.
        
        Parameters:
            value (GPUStatus): GPU metrics snapshot to store as the current status.
        """
        self._gpu_status = value
        self._notify("gpu_status", value)
    
    @property
    def preferences(self) -> UserPreferences:
        """
        Access the current user preferences.
        
        Returns:
            The current UserPreferences instance.
        """
        return self._preferences
    
    # =========================================================================
    # Run Management
    # =========================================================================
    
    def start_run(self, run: RunInfo) -> None:
        """
        Mark the given run as the active running job and notify subscribers.
        
        This updates the provided RunInfo by setting its status to "running", recording the start timestamp, and making it the AppState's current run; subscribers to the "run_started" event are notified with the run.
        
        Parameters:
            run (RunInfo): The run to start; this object will be modified and set as the active run.
        """
        run.status = "running"
        run.started_at = datetime.now().isoformat()
        self._current_run = run
        self._notify("run_started", run)
    
    def update_run_progress(
        self,
        epoch: int,
        loss: float,
        best_loss: Optional[float] = None,
        best_epoch: Optional[int] = None,
    ) -> None:
        """
        Record progress for the active run and update its best-loss metrics when applicable.
        
        Parameters:
            epoch (int): Current epoch number for the active run.
            loss (float): Current loss value for the active run.
            best_loss (Optional[float]): If provided, override the run's recorded best loss.
            best_epoch (Optional[int]): If provided alongside `best_loss`, override the epoch associated with the best loss.
        
        Notes:
            If there is no active run, this function does nothing. When `loss` is lower than the run's stored best loss, the best loss and best epoch are updated automatically.
        """
        if self._current_run is None:
            return
        
        self._current_run.current_epoch = epoch
        self._current_run.current_loss = loss
        
        if best_loss is not None:
            self._current_run.best_loss = best_loss
        if best_epoch is not None:
            self._current_run.best_epoch = best_epoch
        
        # Track best automatically
        if loss < self._current_run.best_loss:
            self._current_run.best_loss = loss
            self._current_run.best_epoch = epoch
        
        self._notify("run_progress", self._current_run)
    
    def complete_run(self, success: bool = True) -> None:
        """
        Complete the current run and persist its result.
        
        If a current run exists, set its status to "completed" when `success` is True or "failed" when False, record the finish timestamp, insert the run at the front of recent runs (keeping at most 20 entries), save history to disk, notify subscribers of "run_completed", and clear the current run.
        
        Parameters:
            success (bool): Whether the run finished successfully (`True`) or failed (`False`).
        """
        if self._current_run is None:
            return
        
        self._current_run.status = "completed" if success else "failed"
        self._current_run.finished_at = datetime.now().isoformat()
        
        # Add to history
        self._recent_runs.insert(0, self._current_run)
        self._recent_runs = self._recent_runs[:20]  # Keep last 20
        self._save_history()
        
        self._notify("run_completed", self._current_run)
        self._current_run = None
    
    def pause_run(self) -> None:
        """
        Mark the current run as paused and notify subscribed listeners.
        
        If there is no active run, this method does nothing.
        """
        if self._current_run is None:
            return
        self._current_run.status = "paused"
        self._notify("run_paused", self._current_run)
    
    def resume_run(self) -> None:
        """
        Resume the currently paused run.
        
        If a run is active, set its status to "running" and notify subscribers with the "run_resumed" event. If there is no current run, the call has no effect.
        """
        if self._current_run is None:
            return
        self._current_run.status = "running"
        self._notify("run_resumed", self._current_run)
    
    # =========================================================================
    # GPU Status
    # =========================================================================
    
    def update_gpu_status(
        self,
        vram_used_gb: float,
        vram_total_gb: float,
        utilization: float = 0.0,
        temperature: float = 0.0,
        power: float = 0.0,
        name: str = "",
    ) -> None:
        """
        Update stored GPU metrics and notify subscribers of the new GPU status.
        
        Parameters:
            vram_used_gb (float): VRAM currently used in gigabytes.
            vram_total_gb (float): Total VRAM in gigabytes.
            utilization (float): GPU utilization percentage (0–100).
            temperature (float): GPU temperature in degrees Celsius.
            power (float): GPU power draw in watts.
            name (str): Optional GPU name or identifier; if provided, replaces the stored name.
        """
        if name:
            self._gpu_status.name = name
        self._gpu_status.vram_used_gb = vram_used_gb
        self._gpu_status.vram_total_gb = vram_total_gb
        self._gpu_status.utilization_percent = utilization
        self._gpu_status.temperature_c = temperature
        self._gpu_status.power_w = power
        self._notify("gpu_status", self._gpu_status)
    
    # =========================================================================
    # Preferences
    # =========================================================================
    
    def update_preferences(self, **kwargs) -> None:
        """
        Update stored user preferences with the provided values and persist the changes.
        
        Only keys that correspond to existing attributes on the UserPreferences object are applied; unknown keys are ignored. After updating, preferences are saved to the configuration file and subscribers of the "preferences" event are notified.
        
        Parameters:
            **kwargs: Mapping of preference names to new values to apply.
        """
        for key, value in kwargs.items():
            if hasattr(self._preferences, key):
                setattr(self._preferences, key, value)
        self._save_config()
        self._notify("preferences", self._preferences)
    
    # =========================================================================
    # Event System
    # =========================================================================
    
    def subscribe(self, event: str, callback: Callable) -> None:
        """
        Register a callback to be invoked when the named event is emitted.
        
        Parameters:
            event (str): The name of the event to subscribe to.
            callback (Callable): A callable that will be called with a single argument containing event data when the event is emitted.
        """
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def unsubscribe(self, event: str, callback: Callable) -> None:
        """
        Remove a previously registered listener callback for a named event.
        
        If the callback is not registered for the given event, this function does nothing.
        
        Parameters:
            event (str): The event name to unsubscribe from.
            callback (Callable): The listener callback to remove.
        """
        if event in self._listeners:
            try:
                self._listeners[event].remove(callback)
            except ValueError:
                pass
    
    def _notify(self, event: str, data: Any) -> None:
        """
        Notify registered listeners for the given event with the provided data.
        
        Calls each callback subscribed to `event` and passes `data` as the single argument. Exceptions raised by any listener are caught and ignored so that one failing listener does not stop notifications to others.
        
        Parameters:
            event (str): The name of the event to notify.
            data (Any): The payload passed to each listener callback.
        """
        for callback in self._listeners.get(event, []):
            try:
                callback(data)
            except Exception:
                pass  # Don't let listener errors crash the app
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def _ensure_config_dir(self) -> None:
        """Ensure the config directory exists."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> None:
        """
        Load persisted user preferences from the config file into this AppState.
        
        If the config file is absent or cannot be read/parsed, the method leaves the current preferences unchanged (defaults remain).
        """
        if not self.CONFIG_FILE.exists():
            return
        
        try:
            with open(self.CONFIG_FILE) as f:
                data = json.load(f)
            self._preferences = UserPreferences.from_dict(data.get("preferences", {}))
        except Exception:
            pass  # Use defaults on error
    
    def _save_config(self) -> None:
        """
        Persist the current user preferences to the configured config file.
        
        Writes the preferences as a JSON object under the "preferences" key to self.CONFIG_FILE and ensures the config directory exists; I/O or serialization errors are suppressed (no exception is raised).
        """
        self._ensure_config_dir()
        try:
            data = {"preferences": self._preferences.to_dict()}
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail
    
    def _load_history(self) -> None:
        """Load run history from disk."""
        if not self.HISTORY_FILE.exists():
            return
        
        try:
            with open(self.HISTORY_FILE) as f:
                data = json.load(f)
            self._recent_runs = [
                RunInfo.from_dict(run) for run in data.get("runs", [])
            ]
        except Exception:
            pass  # Use empty history on error
    
    def _save_history(self) -> None:
        """Save run history to disk."""
        self._ensure_config_dir()
        try:
            data = {"runs": [run.to_dict() for run in self._recent_runs]}
            with open(self.HISTORY_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Collects simple run statistics for dashboard display.
        
        Returns:
            stats (Dict[str, Any]): A dictionary with:
                - total_runs (int): Number of recent runs tracked.
                - completed_runs (int): Count of runs whose status is "completed".
                - failed_runs (int): Count of runs whose status is "failed".
                - has_active_run (bool): `True` if a current run is active, `False` otherwise.
        """
        completed = sum(1 for r in self._recent_runs if r.status == "completed")
        failed = sum(1 for r in self._recent_runs if r.status == "failed")

        return {
            "total_runs": len(self._recent_runs),
            "completed_runs": completed,
            "failed_runs": failed,
            "has_active_run": self._current_run is not None,
        }

    # =========================================================================
    # Estimation Results
    # =========================================================================

    def get_last_estimation_modules(self) -> Optional[List[str]]:
        """
        Retrieve top module names from the most recent estimation file.
        
        Each returned name is taken from an item's "module" field when present, otherwise from its "name" field. If the file is missing, unreadable, or contains invalid data, no names are returned.
        
        Returns:
            List[str]: Module names in the order they appear, or `None` if unavailable or on error.
        """
        if not self._last_estimation:
            return None
        path = Path(self._last_estimation)
        if not path.is_file():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return [
                item.get("module", item.get("name", ""))
                for item in data
                if item.get("module") or item.get("name")
            ]
        except Exception:
            return None

    def save_last_paths(
        self,
        checkpoint_dir: str = "",
        dataset_dir: str = "",
        output_dir: str = "",
    ) -> None:
        """
        Persist provided non-empty directory paths into user preferences so they become defaults for future sessions.
        
        Parameters:
            checkpoint_dir (str): Path to set as the default checkpoint directory; ignored if empty.
            dataset_dir (str): Path to set as the default dataset directory; ignored if empty.
            output_dir (str): Path to set as the default output directory; ignored if empty.
        """
        updates = {}
        if checkpoint_dir:
            updates["default_checkpoint_dir"] = checkpoint_dir
        if dataset_dir:
            updates["default_dataset_dir"] = dataset_dir
        if output_dir:
            updates["default_output_dir"] = output_dir
        if updates:
            self.update_preferences(**updates)
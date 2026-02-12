"""
Settings Screen

Application preferences with:
- Default paths
- UI preferences
- Keyboard shortcuts reference
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Static,
    Button,
    Input,
    Switch,
    Label,
    Rule,
    Select,
    Collapsible,
)
from textual.binding import Binding


class SettingsScreen(Screen):
    """Application settings and preferences."""
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+s", "save", "Save"),
    ]
    
    CSS = """
    SettingsScreen {
        layout: vertical;
    }
    
    #settings-header {
        height: 3;
        background: $panel;
        padding: 0 2;
        layout: horizontal;
        align: left middle;
    }
    
    #settings-title {
        width: 1fr;
        text-style: bold;
        color: $primary;
    }
    
    #settings-controls {
        width: auto;
    }
    
    #settings-content {
        height: 1fr;
        padding: 1 2;
    }
    
    .section {
        margin-bottom: 2;
    }
    
    .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    .setting-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
        align: left middle;
    }
    
    .setting-label {
        width: 30%;
        padding-right: 1;
        color: $text-muted;
    }
    
    .setting-input {
        width: 50%;
    }
    
    .setting-hint {
        width: 20%;
        color: $text-muted;
        padding-left: 1;
    }
    
    Input {
        width: 100%;
    }
    
    Select {
        width: 100%;
    }
    
    #shortcuts-list {
        height: auto;
        padding: 1;
        background: $panel;
        border: round $primary 30%;
    }
    
    .shortcut-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 0;
    }
    
    .shortcut-key {
        width: 15;
        text-style: bold;
        color: $primary;
    }
    
    .shortcut-desc {
        width: 1fr;
    }
    
    #settings-footer {
        height: auto;
        padding: 1 2;
        background: $panel;
    }
    
    #save-status {
        color: $text-muted;
    }
    """
    
    def __init__(self, name: Optional[str] = None) -> None:
        """
        Initialize the SettingsScreen and clear the unsaved-changes flag.
        
        Parameters:
            name (Optional[str]): Optional screen name passed to the base Screen.
        """
        super().__init__(name=name)
        self._unsaved_changes = False
    
    def compose(self) -> ComposeResult:
        """
        Builds and yields the widget tree for the settings screen.
        
        Composes header controls, the scrollable content (Default Paths, UI Preferences, Keyboard Shortcuts, About), and a footer status line, yielding the containers and widgets that make up the SettingsScreen UI.
        
        Returns:
            ComposeResult: An iterable of UI components (containers, widgets, and controls) used to render the settings screen.
        """
        # Header
        with Container(id="settings-header"):
            yield Static("Settings", id="settings-title")
            with Horizontal(id="settings-controls"):
                yield Button("Save", id="btn-save", variant="success")
                yield Button("Reset", id="btn-reset", variant="warning")
                yield Button("Back", id="btn-back", variant="default")
        
        # Main content
        with ScrollableContainer(id="settings-content"):
            # Default Paths
            with Vertical(classes="section"):
                yield Label("Default Paths", classes="section-title")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Checkpoint Directory:", classes="setting-label")
                    yield Input(
                        placeholder="./checkpoints",
                        id="input-default-checkpoint",
                        classes="setting-input",
                    )
                    yield Static("Model files", classes="setting-hint")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Output Directory:", classes="setting-label")
                    yield Input(
                        placeholder="./lora_output",
                        id="input-default-output",
                        classes="setting-input",
                    )
                    yield Static("LoRA weights", classes="setting-hint")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Dataset Directory:", classes="setting-label")
                    yield Input(
                        placeholder="./datasets",
                        id="input-default-dataset",
                        classes="setting-input",
                    )
                    yield Static("Training data", classes="setting-hint")
            
            yield Rule()
            
            # UI Preferences
            with Vertical(classes="section"):
                yield Label("UI Preferences", classes="section-title")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Theme:", classes="setting-label")
                    yield Select(
                        [
                            ("Dark (default)", "dark"),
                            ("Light", "light"),
                            ("System", "system"),
                        ],
                        value="dark",
                        id="select-theme",
                        classes="setting-input",
                    )
                    yield Static("", classes="setting-hint")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Show GPU in Header:", classes="setting-label")
                    with Container(classes="setting-input"):
                        yield Switch(value=True, id="switch-gpu-header")
                    yield Static("Status bar GPU", classes="setting-hint")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Auto-save Config:", classes="setting-label")
                    with Container(classes="setting-input"):
                        yield Switch(value=True, id="switch-auto-save")
                    yield Static("Remember settings", classes="setting-hint")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Confirm on Quit:", classes="setting-label")
                    with Container(classes="setting-input"):
                        yield Switch(value=True, id="switch-confirm-quit")
                    yield Static("Prevent accidents", classes="setting-hint")
            
            yield Rule()
            
            # Keyboard Shortcuts
            with Collapsible(title="Keyboard Shortcuts", collapsed=False):
                with Vertical(id="shortcuts-list"):
                    yield from self._compose_shortcuts()
            
            yield Rule()
            
            # About
            with Vertical(classes="section"):
                yield Label("About", classes="section-title")
                yield Static("Side-Step v0.2.0 by dernet")
                yield Static("ACE-Step LoRA Training Interface")
                yield Static("")
                yield Static("Built with Textual")
        
        # Footer with status
        with Container(id="settings-footer"):
            yield Static(
                f"Settings auto-saved to {self.app.app_state.CONFIG_DIR}" if hasattr(self, "app") and hasattr(self.app, "app_state") else "Settings auto-saved",
                id="save-status",
            )
    
    def _compose_shortcuts(self) -> ComposeResult:
        """
        Builds the keyboard shortcuts UI sections for display in the settings screen.
        
        Yields UI nodes that group shortcuts into labeled sections. Each section yields a bolded header followed by one or more rows; each row contains a key label (class `shortcut-key`) and a description (class `shortcut-desc`). An empty Static line is yielded after each section for spacing.
        
        Returns:
            ComposeResult: A sequence of widget nodes (Static headers and Horizontal rows) suitable for composition into the screen.
        """
        shortcuts = [
            ("Global", [
                ("Q", "Quit application"),
                ("?", "Show help"),
                ("Escape", "Go back / Cancel"),
            ]),
            ("Dashboard", [
                ("F", "New Fixed Training"),
                ("V", "New Vanilla Training"),
                ("P", "Preprocess datasets"),
                ("H", "View run history"),
                ("S", "Open settings"),
            ]),
            ("Training Monitor", [
                ("P", "Pause/Resume training"),
                ("S", "Stop training"),
                ("L", "Toggle log panel"),
            ]),
            ("Navigation", [
                ("Tab", "Next field"),
                ("Shift+Tab", "Previous field"),
                ("Enter", "Select / Confirm"),
                ("↑/↓", "Navigate lists"),
            ]),
        ]
        
        for section, keys in shortcuts:
            yield Static(f"[bold]{section}[/bold]")
            for key, desc in keys:
                with Horizontal(classes="shortcut-row"):
                    yield Static(f"  {key}", classes="shortcut-key")
                    yield Static(desc, classes="shortcut-desc")
            yield Static("")
    
    def on_mount(self) -> None:
        """
        Populate the screen's UI controls with the current application preferences when the screen mounts.
        
        Sets the values of:
        - input-default-checkpoint to the saved default checkpoint directory
        - input-default-output to the saved default output directory
        - input-default-dataset to the saved default dataset directory
        - switch-gpu-header to the saved show-GPU-in-header preference
        - switch-auto-save to the saved auto-save-config preference
        - switch-confirm-quit to the saved confirm-on-quit preference
        """
        prefs = self.app.app_state.preferences
        
        self.query_one("#input-default-checkpoint", Input).value = prefs.default_checkpoint_dir
        self.query_one("#input-default-output", Input).value = prefs.default_output_dir
        self.query_one("#input-default-dataset", Input).value = prefs.default_dataset_dir
        self.query_one("#switch-gpu-header", Switch).value = prefs.show_gpu_in_header
        self.query_one("#switch-auto-save", Switch).value = prefs.auto_save_config
        self.query_one("#switch-confirm-quit", Switch).value = prefs.confirm_on_quit
    
    def _collect_settings(self) -> dict:
        """
        Collect current settings values from the settings UI.
        
        Returns:
            settings (dict): Mapping of preference keys to their current values:
                - default_checkpoint_dir (str): Path for model checkpoint files.
                - default_output_dir (str): Path for LoRA output files.
                - default_dataset_dir (str): Path for dataset files.
                - show_gpu_in_header (bool): Whether to show GPU status in the header.
                - auto_save_config (bool): Whether settings should be auto-saved.
                - confirm_on_quit (bool): Whether to require confirmation when quitting.
        """
        return {
            "default_checkpoint_dir": self.query_one("#input-default-checkpoint", Input).value,
            "default_output_dir": self.query_one("#input-default-output", Input).value,
            "default_dataset_dir": self.query_one("#input-default-dataset", Input).value,
            "show_gpu_in_header": self.query_one("#switch-gpu-header", Switch).value,
            "auto_save_config": self.query_one("#switch-auto-save", Switch).value,
            "confirm_on_quit": self.query_one("#switch-confirm-quit", Switch).value,
        }
    
    def _mark_changed(self) -> None:
        """
        Record that current settings have unsaved changes and update the visible save status.
        
        Sets the internal unsaved-changes flag and updates the save-status widget text to "Unsaved changes".
        """
        self._unsaved_changes = True
        self.query_one("#save-status", Static).update("Unsaved changes")
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Mark as changed when input changes."""
        self._mark_changed()
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """
        Set the screen's unsaved-changes state when a switch value changes.
        """
        self._mark_changed()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """
        Mark the settings as changed when any Select widget value is modified.
        
        Parameters:
            event (Select.Changed): The select change event from the UI.
        """
        self._mark_changed()
    
    # =========================================================================
    # Button Handlers
    # =========================================================================
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Dispatches button press events to the corresponding screen actions.
        
        Parameters:
            event (Button.Pressed): The button press event. Buttons with id "btn-back", "btn-save", and "btn-reset" invoke the back, save, and reset actions respectively.
        """
        if event.button.id == "btn-back":
            self.action_back()
        elif event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-reset":
            self._reset_defaults()
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    def action_back(self) -> None:
        """
        Navigate back from the settings screen.
        
        If there are unsaved changes and the application's preferences enable auto-save, saves the current settings before closing the settings screen. Closes the settings screen.
        """
        if self._unsaved_changes:
            # Auto-save if enabled
            if self.app.app_state.preferences.auto_save_config:
                self.action_save()
        self.app.pop_screen()
    
    def action_save(self) -> None:
        """
        Persist current settings from the UI into the application's preferences.
        
        Collects values from the settings form, updates the application's stored preferences, clears the unsaved-changes flag, updates the save-status display to indicate success, and shows a brief "Settings saved" notification.
        """
        settings = self._collect_settings()
        self.app.app_state.update_preferences(**settings)
        
        self._unsaved_changes = False
        self.query_one("#save-status", Static).update("Settings saved ✓")
        self.notify("Settings saved", timeout=2)
    
    def _reset_defaults(self) -> None:
        """
        Restore the settings UI controls to the application's default preference values.
        
        Sets the default checkpoint, output, and dataset paths and resets the GPU/header, auto-save, and confirm-on-quit toggles; marks the screen as having unsaved changes and shows a brief notification advising to save to apply.
        """
        self.query_one("#input-default-checkpoint", Input).value = "./checkpoints"
        self.query_one("#input-default-output", Input).value = "./lora_output"
        self.query_one("#input-default-dataset", Input).value = "./datasets"
        self.query_one("#switch-gpu-header", Switch).value = True
        self.query_one("#switch-auto-save", Switch).value = True
        self.query_one("#switch-confirm-quit", Switch).value = True
        
        self._mark_changed()
        self.notify("Reset to defaults (save to apply)", timeout=3)
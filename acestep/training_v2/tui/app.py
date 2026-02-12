"""
Side-Step TUI - Main Textual Application

This is the entry point for the TUI interface, handling:
- Screen navigation
- Global keybindings
- Application state coordination
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer

from acestep.training_v2.tui.theme import APP_CSS, register_sidestep_theme
from acestep.training_v2.tui.state import AppState


class SideStepApp(App):
    """
    Side-Step - Interactive TUI for ACE-Step LoRA Training
    
    A full-featured terminal interface for configuring, running,
    and monitoring LoRA fine-tuning jobs.
    """
    
    TITLE = "Side-Step"
    SUB_TITLE = "ACE-Step LoRA Training by dernet"
    CSS = APP_CSS
    
    # Global keybindings
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("?", "help", "Help"),
        Binding("d", "goto_dashboard", "Dashboard", show=True),
        Binding("f", "new_fixed_training", "Fixed Training"),
        Binding("v", "new_vanilla_training", "Vanilla Training"),
        Binding("p", "goto_preprocess", "Preprocess"),
        Binding("e", "goto_estimate", "Estimate"),
        Binding("h", "goto_history", "History"),
        Binding("s", "goto_settings", "Settings"),
        Binding("escape", "back", "Back"),
    ]
    
    def __init__(self) -> None:
        """
        Initialize the SideStepApp, create its application state, and register the Side-Step theme.
        
        Creates an AppState instance assigned to `self.app_state` and registers/applies the custom Side-Step theme for the application.
        """
        super().__init__()
        self.app_state = AppState()
        # Register and apply the custom Side-Step theme
        register_sidestep_theme(self)
    
    def compose(self) -> ComposeResult:
        """Compose the main application layout."""
        yield Header(show_clock=True)
        yield Footer()
    
    def on_mount(self) -> None:
        """
        Push the dashboard screen onto the app's screen stack as the initial screen when the application is mounted.
        
        This method is called during mount and ensures the DashboardScreen becomes the active root screen.
        """
        # Import here to avoid circular imports
        from acestep.training_v2.tui.screens.dashboard import DashboardScreen
        
        # Push the dashboard as the initial screen
        self.push_screen(DashboardScreen())
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    def action_goto_dashboard(self) -> None:
        """
        Navigate to the dashboard screen, removing intermediate screens so the dashboard becomes the active root screen.
        
        If the current screen is not already the dashboard, switch to a new DashboardScreen instance.
        """
        from acestep.training_v2.tui.screens.dashboard import DashboardScreen
        
        # Pop all screens and push dashboard
        while len(self.screen_stack) > 1:
            self.pop_screen()
        
        if not isinstance(self.screen, DashboardScreen):
            self.switch_screen(DashboardScreen())
    
    def action_new_fixed_training(self) -> None:
        """Start a new fixed training configuration."""
        from acestep.training_v2.tui.screens.training_config import TrainingConfigScreen
        
        self.push_screen(TrainingConfigScreen(trainer_type="fixed"))
    
    def action_new_vanilla_training(self) -> None:
        """
        Open the training configuration screen preconfigured for a "vanilla" trainer.
        
        Pushes a new training configuration screen onto the screen stack with the trainer type set to "vanilla".
        """
        from acestep.training_v2.tui.screens.training_config import TrainingConfigScreen
        
        self.push_screen(TrainingConfigScreen(trainer_type="vanilla"))
    
    def action_goto_preprocess(self) -> None:
        """
        Open the dataset browser configured for browsing and preprocessing datasets.
        
        This pushes the DatasetBrowserScreen onto the screen stack with mode="browse".
        """
        from acestep.training_v2.tui.screens.dataset_browser import DatasetBrowserScreen
        
        self.push_screen(DatasetBrowserScreen(mode="browse"))
    
    def action_goto_history(self) -> None:
        """
        Switches the application to the run history screen.
        
        Pushes the run history screen onto the screen stack so it becomes the active view.
        """
        from acestep.training_v2.tui.screens.run_history import RunHistoryScreen
        
        self.push_screen(RunHistoryScreen())
    
    def action_goto_settings(self) -> None:
        """
        Navigate to the Settings screen.
        
        Pushes the SettingsScreen onto the application's screen stack so it becomes the active view.
        """
        from acestep.training_v2.tui.screens.settings import SettingsScreen
        
        self.push_screen(SettingsScreen())
    
    def action_goto_estimate(self) -> None:
        """
        Open the gradient estimation configuration screen.
        
        This pushes the EstimateConfigScreen onto the application's screen stack so the user can configure and run gradient estimation.
        """
        from acestep.training_v2.tui.screens.estimate import EstimateConfigScreen
        
        self.push_screen(EstimateConfigScreen())
    
    def action_help(self) -> None:
        """Show help overlay."""
        help_text = (
            "[bold]Quick Start Workflow:[/bold]\n"
            "1. [P] Preprocess your audio files first\n"
            "2. [E] Estimate to find best layers (optional)\n"
            "3. [F] Fixed Training (recommended) or [V] Vanilla\n\n"
            "[bold]Navigation:[/bold]\n"
            "  [D] Dashboard       [H] History\n"
            "  [S] Settings        [Q] Quit\n"
            "  [Esc] Go Back       [?] This Help\n\n"
            "[bold]Training Types:[/bold]\n"
            "• Fixed = corrected training logic (recommended)\n"
            "• Vanilla = original behavior (for compatibility)\n\n"
            "[bold]Tips:[/bold]\n"
            "• Tab/Shift+Tab to navigate forms\n"
            "• Each option shows impact hints below it"
        )
        self.notify(help_text, title="Side-Step v0.2.0 -- Help", timeout=20)
    
    def action_back(self) -> None:
        """Go back to previous screen."""
        if len(self.screen_stack) > 1:
            self.pop_screen()


def main() -> None:
    """
    Start and run the Side-Step TUI application.
    
    Initializes a SideStepApp instance and runs its main event loop.
    """
    app = SideStepApp()
    app.run()


if __name__ == "__main__":
    main()
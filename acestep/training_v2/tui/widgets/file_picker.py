"""
File/Directory Picker Widget

Interactive file browser with:
- Keyboard navigation (j/k, enter, backspace)
- Directory tree view
- File filtering by extension
- Quick path input
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Input, Button, DirectoryTree, Tree
from textual.widget import Widget
from textual.binding import Binding
from textual.message import Message
from textual.screen import ModalScreen


class FilePicker(Widget):
    """
    File/directory picker widget with tree navigation.
    
    Can be used inline or as a modal dialog.
    """
    
    DEFAULT_CSS = """
    FilePicker {
        height: 100%;
        border: round $primary 35%;
        padding: 1;
        layout: vertical;
    }
    
    #picker-header {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }
    
    #picker-path-input {
        width: 1fr;
    }
    
    #picker-path-input Input {
        width: 100%;
    }
    
    #picker-up-btn {
        width: auto;
        margin-left: 1;
    }
    
    #picker-tree {
        height: 1fr;
        border: round $primary 25%;
    }
    
    #picker-footer {
        height: 3;
        layout: horizontal;
        align: right middle;
    }
    
    #picker-footer Button {
        margin-left: 1;
    }
    
    DirectoryTree {
        height: 100%;
    }
    """
    
    BINDINGS = [
        Binding("enter", "select", "Select", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("backspace", "parent", "Parent Dir"),
        Binding("h", "parent", "Parent Dir", show=False),
    ]
    
    class Selected(Message):
        """Message sent when a path is selected."""
        def __init__(self, path: Path) -> None:
            """
            Initialize the widget with a starting filesystem path.
            
            Parameters:
                path (Path): The initial filesystem path to display and navigate from.
            """
            super().__init__()
            self.path = path
    
    class Cancelled(Message):
        """Message sent when selection is cancelled."""
        pass
    
    def __init__(
        self,
        start_path: Optional[Path] = None,
        select_directory: bool = True,
        file_filter: Optional[str] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        """
        Create a FilePicker configured for a starting path, selection mode, and optional file filter.
        
        Parameters:
            start_path: Path to open initially; defaults to the current working directory if not provided.
            select_directory: Whether selections should target directories; when False, files are selectable.
            file_filter: Optional glob pattern to restrict visible files (for example, "*.pt").
            name: Optional widget name.
            id: Optional widget identifier.
            classes: Optional CSS classes to apply to the widget.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.start_path = start_path or Path.cwd()
        self.select_directory = select_directory
        self.file_filter = file_filter
        self._current_path = self.start_path
    
    def compose(self) -> ComposeResult:
        """
        Build the widget tree for the file picker UI.
        
        Returns:
            compose_result (ComposeResult): Sequence of widgets and layout containers that form the file picker interface.
        """
        # Path input
        with Horizontal(id="picker-header"):
            with Container(id="picker-path-input"):
                yield Input(value=str(self._current_path), id="path-input")
            yield Button("↑", id="picker-up-btn", variant="default")
        
        # Directory tree
        with Container(id="picker-tree"):
            yield DirectoryTree(self._current_path, id="dir-tree")
        
        # Footer with action buttons
        with Horizontal(id="picker-footer"):
            yield Button("Cancel", id="btn-cancel", variant="default")
            yield Button("Select", id="btn-select", variant="primary")
    
    def on_mount(self) -> None:
        """
        Focuses the directory tree widget when the FilePicker is mounted.
        
        This moves keyboard focus to the directory tree so it is ready to receive navigation input.
        """
        tree = self.query_one("#dir-tree", DirectoryTree)
        tree.focus()
    
    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        """
        Update picker state when a directory is selected in the directory tree.
        
        Updates the widget's current path and synchronizes the path input field with the selected directory.
        
        Parameters:
            event (DirectoryTree.DirectorySelected): Event carrying the selected directory path in `event.path`.
        """
        self._current_path = event.path
        self.query_one("#path-input", Input).value = str(event.path)
    
    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        """
        Update the picker's current path and path-input when a file is selected in the directory tree.
        
        If the picker is configured to select files (not directories), sets the widget's current path to the selected file and updates the "#path-input" Input value to the file's path.
        
        Parameters:
            event (DirectoryTree.FileSelected): Event containing the selected file path in `event.path`.
        """
        if not self.select_directory:
            self._current_path = event.path
            self.query_one("#path-input", Input).value = str(event.path)
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Update the picker to the submitted path if it exists.
        
        When the input with id "path-input" is submitted, validate that the entered path exists; if so, set the widget's current path to that Path, update the directory tree's path, and reload the tree.
        
        Parameters:
            event (Input.Submitted): The submission event from the path input field.
        """
        if event.input.id == "path-input":
            path = Path(event.value)
            if path.exists():
                self._current_path = path
                tree = self.query_one("#dir-tree", DirectoryTree)
                tree.path = path
                tree.reload()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Route button-press events to the picker's actions.
        
        Recognizes these button IDs:
        - "btn-select": trigger selection.
        - "btn-cancel": trigger cancellation.
        - "picker-up-btn": navigate to the parent directory.
        
        Parameters:
            event (Button.Pressed): The button press event whose button id determines the action.
        """
        if event.button.id == "btn-select":
            self.action_select()
        elif event.button.id == "btn-cancel":
            self.action_cancel()
        elif event.button.id == "picker-up-btn":
            self.action_parent()
    
    def action_select(self) -> None:
        """Select the current path."""
        self.post_message(self.Selected(self._current_path))
    
    def action_cancel(self) -> None:
        """Cancel selection."""
        self.post_message(self.Cancelled())
    
    def action_parent(self) -> None:
        """
        Navigate to the parent directory of the current path if it exists.
        
        If the parent directory exists, updates the widget's current path, sets the path input value to the parent, updates the DirectoryTree's path, and reloads the tree.
        """
        parent = self._current_path.parent
        if parent.exists():
            self._current_path = parent
            self.query_one("#path-input", Input).value = str(parent)
            tree = self.query_one("#dir-tree", DirectoryTree)
            tree.path = parent
            tree.reload()


class FilePickerModal(ModalScreen[Optional[Path]]):
    """
    Modal dialog for file/directory selection.
    
    Usage:
        path = await self.app.push_screen_wait(FilePickerModal())
    """
    
    DEFAULT_CSS = """
    FilePickerModal {
        align: center middle;
    }
    
    #picker-container {
        width: 80%;
        height: 80%;
        max-width: 100;
        max-height: 40;
        border: round $primary;
        background: $surface;
    }
    
    #picker-title {
        dock: top;
        height: 3;
        padding: 1 2;
        background: $panel;
        text-style: bold;
    }
    
    #picker-content {
        height: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]
    
    def __init__(
        self,
        title: str = "Select Path",
        start_path: Optional[Path] = None,
        select_directory: bool = True,
        file_filter: Optional[str] = None,
    ) -> None:
        """
        Initialize the FilePickerModal with display text, initial path, selection mode, and an optional file filter.
        
        Parameters:
            title (str): Title text shown at the top of the modal.
            start_path (Optional[Path]): Initial filesystem path to show in the embedded FilePicker; if None, defaults to the current working directory.
            select_directory (bool): If True, the picker selects directories; if False, the picker selects files.
            file_filter (Optional[str]): Optional filename glob or extension filter applied by the embedded FilePicker (e.g., "*.txt" or ".py").
        """
        super().__init__()
        self.title_text = title
        self.start_path = start_path
        self.select_directory = select_directory
        self.file_filter = file_filter
    
    def compose(self) -> ComposeResult:
        """
        Compose the modal's widget tree for the file picker.
        
        Returns:
            ComposeResult: Yields a container with the modal title and an inner FilePicker configured with the modal's title, start path, selection mode, and file filter.
        """
        with Container(id="picker-container"):
            yield Static(self.title_text, id="picker-title")
            with Container(id="picker-content"):
                yield FilePicker(
                    start_path=self.start_path,
                    select_directory=self.select_directory,
                    file_filter=self.file_filter,
                    id="modal-picker",
                )
    
    def on_file_picker_selected(self, event: FilePicker.Selected) -> None:
        """
        Dismiss the modal using the path chosen in the picker.
        
        Parameters:
            event (FilePicker.Selected): Message containing the selected Path which will be returned when the modal is dismissed.
        """
        self.dismiss(event.path)
    
    def on_file_picker_cancelled(self, event: FilePicker.Cancelled) -> None:
        """
        Dismiss the modal with `None` to indicate the selection was cancelled.
        """
        self.dismiss(None)
    
    def action_cancel(self) -> None:
        """
        Cancel the picker and dismiss the dialog without producing a selection.
        """
        self.dismiss(None)
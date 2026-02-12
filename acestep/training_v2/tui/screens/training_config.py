"""
Training Configuration Screen

Interactive form for configuring training runs with:
- Tabbed interface for different setting categories
- Real-time validation
- VRAM estimation preview
- File/directory browsing
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
    Label,
    Select,
    Switch,
    Rule,
    TabbedContent,
    TabPane,
    RadioButton,
    RadioSet,
)
from textual.binding import Binding
from textual.validation import Number, Length

from rich.text import Text


class TrainingConfigScreen(Screen):
    """Configuration screen for setting up training runs."""
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "start_training", "Start Training"),
        Binding("ctrl+r", "reset", "Reset Form"),
        Binding("ctrl+e", "toggle_expert", "Toggle Expert Mode"),
    ]
    
    CSS = """
    TrainingConfigScreen {
        layout: vertical;
    }
    
    #config-header {
        height: 3;
        background: $panel;
        padding: 0 2;
        layout: horizontal;
        align: center middle;
    }
    
    #config-title {
        width: 1fr;
        text-style: bold;
        color: $primary;
    }
    
    #header-buttons {
        width: auto;
    }
    
    #header-buttons Button {
        margin-left: 1;
    }
    
    #config-content {
        height: 1fr;
        padding: 1;
    }
    
    TabbedContent {
        height: 100%;
    }
    
    TabPane {
        padding: 1;
    }
    
    .form-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
        align: left middle;
    }
    
    .form-label {
        width: 25;
        padding-right: 1;
        color: $text-muted;
    }
    
    .form-input {
        width: 1fr;
    }
    
    .form-input Input {
        width: 100%;
    }
    
    .form-input Select {
        width: 100%;
    }
    
    .form-hint {
        color: $text-muted;
        margin-left: 25;
        margin-top: 0;
    }
    
    #preview-panel {
        height: auto;
        min-height: 6;
        border: round $primary 35%;
        padding: 1;
        margin-top: 1;
    }
    
    .preview-title {
        text-style: bold;
        color: $primary;
    }
    
    .preview-row {
        height: auto;
    }
    
    #validation-status {
        height: 2;
        padding: 0 1;
        background: $panel;
    }
    
    .error-text {
        color: $error;
    }
    
    .success-text {
        color: $success;
    }
    
    .section-header {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .expert-section {
        border: round $warning 30%;
        padding: 1;
        margin-top: 1;
    }
    
    .expert-warning {
        color: $warning;
        text-style: italic;
        margin-bottom: 1;
    }
    
    .hidden {
        display: none;
    }
    
    .info-panel {
        border: round $secondary 25%;
        padding: 1;
        margin-bottom: 1;
        background: $panel;
    }
    
    .impact-hint {
        color: $text-muted;
        margin-left: 25;
        margin-top: 0;
        text-style: italic;
    }
    
    RadioSet {
        layout: horizontal;
        height: auto;
        width: 1fr;
    }
    
    RadioButton {
        margin-right: 2;
    }
    """
    
    def __init__(self, trainer_type: str = "fixed") -> None:
        """
        Initialize the training configuration screen for the given trainer type.
        
        Parameters:
        	trainer_type (str): Trainer variant to configure; expected values include "fixed" (default) or "vanilla".
        
        Details:
        	Initializes internal state including an empty validation errors list and expert mode disabled by default.
        """
        super().__init__()
        self.trainer_type = trainer_type
        self._validation_errors: list[str] = []
        self._expert_mode = False
    
    def compose(self) -> ComposeResult:
        """
        Builds and yields the UI tree for the training configuration screen.
        
        Creates the header (title and Start/Cancel buttons), the main scrollable tabbed content populated by the Required, LoRA, Training, Advanced, and Logging tab composers, a configuration preview panel, and the validation status bar.
        
        Returns:
            ComposeResult: The sequence of UI components for this screen.
        """
        # Header
        with Container(id="config-header"):
            title = "Fixed Training" if self.trainer_type == "fixed" else "Vanilla Training"
            yield Static(f"New {title} Run", id="config-title")
            with Horizontal(id="header-buttons"):
                yield Button("Start", id="btn-start", variant="success")
                yield Button("Cancel", id="btn-cancel", variant="error")
        
        # Main content with tabs
        with ScrollableContainer(id="config-content"):
            with TabbedContent():
                # Required Settings Tab
                with TabPane("Required", id="tab-required"):
                    yield from self._compose_required_tab()
                
                # LoRA Settings Tab
                with TabPane("LoRA", id="tab-lora"):
                    yield from self._compose_lora_tab()
                
                # Training Settings Tab
                with TabPane("Training", id="tab-training"):
                    yield from self._compose_training_tab()
                
                # Advanced Settings Tab
                with TabPane("Advanced", id="tab-advanced"):
                    yield from self._compose_advanced_tab()
                
                # Logging Settings Tab
                with TabPane("Logging", id="tab-logging"):
                    yield from self._compose_logging_tab()
            
            # Preview panel
            with Container(id="preview-panel"):
                yield Static("Configuration Preview", classes="preview-title")
                yield Static("", id="preview-content", classes="preview-row")
        
        # Validation status bar
        yield Static("Ready to configure", id="validation-status")
    
    def _compose_required_tab(self) -> ComposeResult:
        """
        Builds the "Required" tab content for the training configuration UI.
        
        Provides UI elements for essential configuration: an info panel describing the selected trainer type, a checkpoint directory input with browse button, a model variant radio group, a dataset directory input with browse button, and an output directory input with browse button; each field includes a short hint.
        
        Returns:
            ComposeResult: An iterable of Textual widgets composing the tab.
        """
        # Info panel for this tab
        yield Static(
            f"[bold]{'Fixed' if self.trainer_type == 'fixed' else 'Vanilla'} Training[/bold]\n\n"
            + (
                "[green]Fixed[/green] uses corrected training logic:\n"
                "• Continuous timestep sampling (matches inference)\n"
                "• CFG dropout for better conditioning\n"
                "→ Recommended for new training runs"
                if self.trainer_type == "fixed" else
                "[yellow]Vanilla[/yellow] reproduces the original training:\n"
                "• Discrete timestep sampling (legacy behavior)\n"
                "• No CFG dropout\n"
                "→ Use only if you need backward compatibility"
            ),
            id="info-trainer-type",
            classes="info-panel",
        )
        
        yield Static("Essential Settings", classes="section-header")
        
        # Checkpoint directory
        with Horizontal(classes="form-row"):
            yield Static("Checkpoint Directory:", classes="form-label")
            with Horizontal(classes="form-input"):
                yield Input(
                    placeholder="./checkpoints/acestep-v15-turbo",
                    id="input-checkpoint-dir",
                )
                yield Button("...", id="btn-browse-checkpoint", variant="default")
        yield Static("Path to model checkpoints (with config.json)", classes="form-hint")
        
        # Model variant
        with Horizontal(classes="form-row"):
            yield Static("Model Variant:", classes="form-label")
            with RadioSet(id="radio-variant"):
                yield RadioButton("Turbo", value=True, id="variant-turbo")
                yield RadioButton("Base", id="variant-base")
                yield RadioButton("SFT", id="variant-sft")
        yield Static("turbo=fastest, base=quality, sft=sing/rap", classes="form-hint")
        
        # Dataset directory
        with Horizontal(classes="form-row"):
            yield Static("Dataset Directory:", classes="form-label")
            with Horizontal(classes="form-input"):
                yield Input(
                    placeholder="./datasets/my_preprocessed",
                    id="input-dataset-dir",
                )
                yield Button("...", id="btn-browse-dataset", variant="default")
        yield Static("Directory containing preprocessed .pt files", classes="form-hint")
        
        # Output directory
        with Horizontal(classes="form-row"):
            yield Static("Output Directory:", classes="form-label")
            with Horizontal(classes="form-input"):
                yield Input(
                    placeholder="./lora_output/my_lora",
                    id="input-output-dir",
                )
                yield Button("...", id="btn-browse-output", variant="default")
        yield Static("Where to save LoRA weights and checkpoints", classes="form-hint")
    
    def _compose_lora_tab(self) -> ComposeResult:
        """
        Builds the LoRA settings tab containing controls for LoRA capacity, attention targeting, and target projection selection.
        
        Provides inputs for rank, alpha, and dropout; a selector for which attention types to train; and a comma-separated field for projection target modules, along with explanatory hints and section headers.
        
        Returns:
            A sequence of Textual widgets composing the LoRA configuration tab.
        """
        # Info panel
        yield Static(
            "[bold]LoRA Configuration[/bold]\n\n"
            "LoRA (Low-Rank Adaptation) adds small trainable matrices to the model.\n"
            "These settings control the capacity and scope of what gets trained.",
            classes="info-panel",
        )
        
        yield Static("Capacity Settings", classes="section-header")
        
        # Rank
        with Horizontal(classes="form-row"):
            yield Static("Rank:", classes="form-label")
            yield Input(
                value="64",
                id="input-rank",
                validators=[Number(minimum=1, maximum=512)],
                classes="form-input",
            )
        yield Static("Controls LoRA capacity (how much it can learn)", classes="form-hint")
        yield Static("↓ Lower = faster, less VRAM, may underfit  |  ↑ Higher = slower, more VRAM, captures more detail", classes="impact-hint")
        
        # Alpha
        with Horizontal(classes="form-row"):
            yield Static("Alpha:", classes="form-label")
            yield Input(
                value="128",
                id="input-alpha",
                validators=[Number(minimum=1, maximum=1024)],
                classes="form-input",
            )
        yield Static("Scaling factor (usually 1-2× the rank)", classes="form-hint")
        yield Static("↓ Lower = subtler effect  |  ↑ Higher = stronger effect on output", classes="impact-hint")
        
        # Dropout
        with Horizontal(classes="form-row"):
            yield Static("Dropout:", classes="form-label")
            yield Input(
                value="0.0",
                id="input-dropout",
                validators=[Number(minimum=0.0, maximum=1.0)],
                classes="form-input",
            )
        yield Static("Regularization to prevent overfitting", classes="form-hint")
        yield Static("0 = no dropout  |  ↑ Higher = more regularization, may need more epochs", classes="impact-hint")
        
        yield Rule()
        yield Static("Attention Targeting", classes="section-header")
        
        # Attention type
        with Horizontal(classes="form-row"):
            yield Static("Attention Type:", classes="form-label")
            yield Select(
                [
                    ("Both (Self + Cross)", "both"),
                    ("Self-Attention Only", "self"),
                    ("Cross-Attention Only", "cross"),
                ],
                value="both",
                id="select-attention-type",
            )
        yield Static("Which attention layers to train", classes="form-hint")
        yield Static("Self = how audio relates to itself  |  Cross = how audio relates to text/lyrics", classes="impact-hint")
        
        # Target modules
        with Horizontal(classes="form-row"):
            yield Static("Target Projections:", classes="form-label")
            yield Input(
                value="to_q,to_k,to_v,to_out.0",
                id="input-target-modules",
                classes="form-input",
            )
        yield Static("Projection layers within attention (comma-separated)", classes="form-hint")
        yield Static("More projections = more trainable params, more VRAM, finer control", classes="impact-hint")
    
    def _compose_training_tab(self) -> ComposeResult:
        """
        Compose the "Training" tab UI with controls for VRAM profile, core training parameters, and learning rate schedule.
        
        The produced tab includes:
        - VRAM profile quick-select and hints.
        - Core parameters: epochs, batch size, gradient accumulation, learning rate, and random seed.
        - Learning rate scheduler and warmup steps.
        
        Returns:
            ComposeResult: A sequence of Textual widgets (Static, Select, Input, Rule, etc.) that make up the Training settings tab. 
        """
        # Info panel
        yield Static(
            "[bold]Training Dynamics[/bold]\n\n"
            "These settings control how the model learns from your data.\n"
            "The balance between epochs, batch size, and learning rate affects convergence.",
            classes="info-panel",
        )
        
        # VRAM Profile Quick Select
        yield Static("VRAM Profile", classes="section-header")
        with Horizontal(classes="form-row"):
            yield Static("VRAM Profile:", classes="form-label")
            yield Select(
                [
                    ("Auto (detect)", "auto"),
                    ("Comfortable (24GB+)", "comfortable"),
                    ("Standard (16-24GB)", "standard"),
                    ("Tight (10-16GB)", "tight"),
                    ("Minimal (<10GB)", "minimal"),
                ],
                value="auto",
                id="select-vram-profile",
            )
        yield Static("Pre-fills recommended settings for your GPU's VRAM. You can override any value.", classes="form-hint")
        yield Static("Higher profiles = more headroom, faster training  |  Lower = aggressive memory savings", classes="impact-hint")
        
        yield Rule()
        yield Static("Core Parameters", classes="section-header")
        
        # Epochs
        with Horizontal(classes="form-row"):
            yield Static("Epochs:", classes="form-label")
            yield Input(
                value="100",
                id="input-epochs",
                validators=[Number(minimum=1, maximum=10000)],
                classes="form-input",
            )
        yield Static("How many times to iterate over your dataset", classes="form-hint")
        yield Static("↓ Fewer = faster, may underfit  |  ↑ More = slower, risk of overfitting", classes="impact-hint")
        
        # Batch size
        with Horizontal(classes="form-row"):
            yield Static("Batch Size:", classes="form-label")
            yield Input(
                value="1",
                id="input-batch-size",
                validators=[Number(minimum=1, maximum=64)],
                classes="form-input",
            )
        yield Static("Samples processed together (limited by VRAM)", classes="form-hint")
        yield Static("↓ Smaller = less VRAM, noisier gradients  |  ↑ Larger = more VRAM, smoother training", classes="impact-hint")
        
        # Gradient accumulation
        with Horizontal(classes="form-row"):
            yield Static("Gradient Accumulation:", classes="form-label")
            yield Input(
                value="4",
                id="input-grad-accum",
                validators=[Number(minimum=1, maximum=128)],
                classes="form-input",
            )
        yield Static("Simulate larger batches without extra VRAM", classes="form-hint")
        yield Static("Effective batch = batch_size × accumulation steps", classes="impact-hint")
        
        # Learning rate
        with Horizontal(classes="form-row"):
            yield Static("Learning Rate:", classes="form-label")
            yield Input(
                value="1e-4",
                id="input-lr",
                classes="form-input",
            )
        yield Static("How aggressively the model updates", classes="form-hint")
        yield Static("↓ Lower = slower, more stable  |  ↑ Higher = faster, may overshoot/diverge", classes="impact-hint")
        
        # Seed
        with Horizontal(classes="form-row"):
            yield Static("Random Seed:", classes="form-label")
            yield Input(
                value="42",
                id="input-seed",
                validators=[Number(minimum=0, maximum=2147483647)],
                classes="form-input",
            )
        yield Static("For reproducible results (same seed = same random choices)", classes="form-hint")
        
        yield Rule()
        yield Static("Learning Rate Schedule", classes="section-header")
        
        # Scheduler
        with Horizontal(classes="form-row"):
            yield Static("LR Scheduler:", classes="form-label")
            yield Select(
                [
                    ("Cosine Annealing", "cosine"),
                    ("Linear", "linear"),
                    ("Constant", "constant"),
                    ("Constant with Warmup", "constant_with_warmup"),
                ],
                value="cosine",
                id="select-scheduler",
            )
        yield Static("How learning rate changes over training", classes="form-hint")
        yield Static("Cosine = gradual decay to zero  |  Linear = steady decay  |  Constant = no decay", classes="impact-hint")
        
        # Warmup steps
        with Horizontal(classes="form-row"):
            yield Static("Warmup Steps:", classes="form-label")
            yield Input(
                value="500",
                id="input-warmup",
                validators=[Number(minimum=0, maximum=10000)],
                classes="form-input",
            )
        yield Static("Gradually increase LR at start to stabilize early training", classes="form-hint")
        yield Static("↓ Fewer = faster start, may be unstable  |  ↑ More = gentler start, slower overall", classes="impact-hint")
    
    def _compose_advanced_tab(self) -> ComposeResult:
        """
        Create the Advanced settings tab UI used to configure device, precision, optimizer, VRAM-saving options, data loading, and expert parameters.
        
        The tab includes controls for device selection, numerical precision, regularization (weight decay and gradient clipping), an optional CFG dropout ratio when using the fixed trainer, optimizer choice, VRAM saving toggles (gradient checkpointing and encoder offload), data loading options (num workers and pin memory), and an expert-mode section with additional low-level controls (bias mode, prefetch factor, persistent workers, and per-layer logging frequency). The expert settings container is hidden by default.
        
        Returns:
            ComposeResult: An iterable of Textual widgets that make up the Advanced settings tab.
        """
        # Info panel
        yield Static(
            "[bold]Advanced Settings[/bold]\n\n"
            "These settings affect performance and stability.\n"
            "Defaults work well for most cases - change with care.",
            classes="info-panel",
        )
        
        yield Static("Device & Precision", classes="section-header")
        
        # Device
        with Horizontal(classes="form-row"):
            yield Static("Device:", classes="form-label")
            yield Select(
                [
                    ("Auto-detect", "auto"),
                    ("CUDA (NVIDIA)", "cuda"),
                    ("MPS (Apple)", "mps"),
                    ("CPU", "cpu"),
                ],
                value="auto",
                id="select-device",
            )
        yield Static("Where to run training (auto-detect recommended)", classes="form-hint")
        
        # Precision
        with Horizontal(classes="form-row"):
            yield Static("Precision:", classes="form-label")
            yield Select(
                [
                    ("BF16 (recommended)", "bf16-mixed"),
                    ("FP16", "16-mixed"),
                    ("FP32", "32-true"),
                ],
                value="bf16-mixed",
                id="select-precision",
            )
        yield Static("Numerical precision for training", classes="form-hint")
        yield Static("BF16 = fast + stable  |  FP16 = fast, may overflow  |  FP32 = slow, most precise", classes="impact-hint")
        
        yield Rule()
        yield Static("Regularization", classes="section-header")
        
        # Weight decay
        with Horizontal(classes="form-row"):
            yield Static("Weight Decay:", classes="form-label")
            yield Input(
                value="0.01",
                id="input-weight-decay",
                classes="form-input",
            )
        yield Static("Penalizes large weights to prevent overfitting", classes="form-hint")
        yield Static("↓ Lower = less regularization  |  ↑ Higher = stronger regularization", classes="impact-hint")
        
        # Max grad norm
        with Horizontal(classes="form-row"):
            yield Static("Max Grad Norm:", classes="form-label")
            yield Input(
                value="1.0",
                id="input-max-grad-norm",
                classes="form-input",
            )
        yield Static("Clips gradients to prevent explosion", classes="form-hint")
        yield Static("↓ Lower = more clipping, slower learning  |  ↑ Higher = less clipping, may be unstable", classes="impact-hint")
        
        # CFG Ratio (only for fixed trainer)
        if self.trainer_type == "fixed":
            with Horizontal(classes="form-row"):
                yield Static("CFG Dropout Ratio:", classes="form-label")
                yield Input(
                    value="0.15",
                    id="input-cfg-ratio",
                    classes="form-input",
                )
            yield Static("Randomly drops conditioning to improve CFG at inference", classes="form-hint")
            yield Static("↓ Lower = relies more on conditioning  |  ↑ Higher = better unconditional generation", classes="impact-hint")
        
        yield Rule()
        yield Static("Optimizer", classes="section-header")
        
        with Horizontal(classes="form-row"):
            yield Static("Optimizer:", classes="form-label")
            yield Select(
                [
                    ("AdamW (default)", "adamw"),
                    ("AdamW 8-bit (lower VRAM)", "adamw8bit"),
                    ("Adafactor (minimal state)", "adafactor"),
                    ("Prodigy (auto-tunes LR)", "prodigy"),
                ],
                value="adamw",
                id="select-optimizer",
            )
        yield Static("Which optimizer to use for weight updates", classes="form-hint")
        yield Static(
            "AdamW = reliable default  |  8-bit = saves ~30% optimizer VRAM\n"
            "Adafactor = extreme VRAM savings  |  Prodigy = auto-tunes LR (set LR=1.0)",
            classes="impact-hint",
        )
        
        yield Rule()
        yield Static("VRAM Savings", classes="section-header")
        
        # Gradient checkpointing
        with Horizontal(classes="form-row"):
            yield Static("Gradient Checkpointing:", classes="form-label")
            yield Switch(value=False, id="switch-grad-ckpt")
        yield Static("Recompute activations during backward pass to save VRAM", classes="form-hint")
        yield Static("OFF = faster training  |  ON = ~40-60% less activation VRAM, ~30% slower", classes="impact-hint")
        
        # Encoder offloading
        with Horizontal(classes="form-row"):
            yield Static("Offload Encoder to CPU:", classes="form-label")
            yield Switch(value=False, id="switch-offload-encoder")
        yield Static("Frees ~2-4GB VRAM by moving unused model parts to CPU after setup", classes="form-hint")
        yield Static("OFF = all on GPU  |  ON = encoder/VAE on CPU (saves VRAM, minimal speed impact)", classes="impact-hint")
        
        yield Rule()
        yield Static("Data Loading", classes="section-header")
        
        # Num workers
        with Horizontal(classes="form-row"):
            yield Static("Num Workers:", classes="form-label")
            yield Input(
                value="4",
                id="input-num-workers",
                validators=[Number(minimum=0, maximum=32)],
                classes="form-input",
            )
        yield Static("Parallel processes for loading data", classes="form-hint")
        yield Static("0 = main thread only  |  ↑ More = faster loading, uses more CPU/RAM", classes="impact-hint")
        
        # Pin memory
        with Horizontal(classes="form-row"):
            yield Static("Pin Memory:", classes="form-label")
            yield Switch(value=True, id="switch-pin-memory")
        yield Static("Pre-allocate GPU transfer memory (faster but uses RAM)", classes="form-hint")
        
        yield Rule()
        
        # Expert Mode Toggle
        with Horizontal(classes="form-row"):
            yield Static("Expert Mode:", classes="form-label")
            yield Switch(value=False, id="switch-expert-mode")
        yield Static("Reveal additional low-level settings", classes="form-hint")
        
        # Expert settings container (hidden by default)
        with Container(id="expert-settings", classes="expert-section hidden"):
            yield Static("⚠️ These settings rarely need changing and can cause issues if misconfigured", classes="expert-warning")
            
            # Bias mode
            with Horizontal(classes="form-row"):
                yield Static("LoRA Bias:", classes="form-label")
                yield Select(
                    [
                        ("None (default)", "none"),
                        ("All biases", "all"),
                        ("LoRA layers only", "lora_only"),
                    ],
                    value="none",
                    id="select-bias",
                )
            yield Static("Whether to also train bias parameters", classes="form-hint")
            yield Static("None = only LoRA weights  |  All = everything  |  LoRA only = biases in LoRA layers", classes="impact-hint")
            
            # Prefetch factor
            with Horizontal(classes="form-row"):
                yield Static("Prefetch Factor:", classes="form-label")
                yield Input(
                    value="2",
                    id="input-prefetch-factor",
                    validators=[Number(minimum=1, maximum=16)],
                    classes="form-input",
                )
            yield Static("DataLoader prefetch multiplier (default: 2)", classes="form-hint")
            
            # Persistent workers
            with Horizontal(classes="form-row"):
                yield Static("Persistent Workers:", classes="form-label")
                yield Switch(value=True, id="switch-persistent-workers")
            yield Static("Keep DataLoader workers alive between epochs", classes="form-hint")
            
            # Log heavy every
            with Horizontal(classes="form-row"):
                yield Static("Log Heavy Every:", classes="form-label")
                yield Input(
                    value="50",
                    id="input-log-heavy-every",
                    validators=[Number(minimum=1, maximum=1000)],
                    classes="form-input",
                )
            yield Static("Steps between per-layer gradient norm logging", classes="form-hint")
    
    def _compose_logging_tab(self) -> ComposeResult:
        """
        Builds the "Logging" tab UI for configuring TensorBoard logging, checkpointing, resume options, and audio sampling.
        
        Composes labeled form rows and helper hints for:
        - TensorBoard log directory and log frequency.
        - Checkpointing frequency and retention (save every N epochs, keep last N).
        - Optional resume-from checkpoint path with file picker.
        - Periodic audio sampling frequency and related guidance.
        
        Returns:
            ComposeResult: A sequence of Textual widgets that make up the Logging tab.
        """
        # Info panel
        yield Static(
            "[bold]Monitoring & Checkpoints[/bold]\n\n"
            "Track progress with TensorBoard and save checkpoints to resume later.\n"
            "More frequent saves = safer but uses more disk space.",
            classes="info-panel",
        )
        
        yield Static("TensorBoard Logging", classes="section-header")
        
        # Log directory
        with Horizontal(classes="form-row"):
            yield Static("Log Directory:", classes="form-label")
            with Horizontal(classes="form-input"):
                yield Input(
                    placeholder="./logs",
                    id="input-log-dir",
                )
                yield Button("...", id="btn-browse-logs", variant="default")
        yield Static("Where to save TensorBoard logs (view with: tensorboard --logdir ./logs)", classes="form-hint")
        
        # Log every N steps
        with Horizontal(classes="form-row"):
            yield Static("Log Every N Steps:", classes="form-label")
            yield Input(
                value="10",
                id="input-log-every",
                validators=[Number(minimum=1, maximum=1000)],
                classes="form-input",
            )
        yield Static("How often to record metrics", classes="form-hint")
        yield Static("↓ More frequent = detailed graphs, slight overhead  |  ↑ Less frequent = lighter logging", classes="impact-hint")
        
        yield Rule()
        yield Static("Checkpointing", classes="section-header")
        
        # Save every N epochs
        with Horizontal(classes="form-row"):
            yield Static("Save Every N Epochs:", classes="form-label")
            yield Input(
                value="10",
                id="input-save-every",
                validators=[Number(minimum=1, maximum=1000)],
                classes="form-input",
            )
        yield Static("How often to save a checkpoint", classes="form-hint")
        yield Static("↓ More frequent = safer, more disk  |  ↑ Less frequent = less disk, more to redo if crash", classes="impact-hint")
        
        # Keep last N checkpoints
        with Horizontal(classes="form-row"):
            yield Static("Keep Last N:", classes="form-label")
            yield Input(
                value="3",
                id="input-keep-last",
                validators=[Number(minimum=1, maximum=100)],
                classes="form-input",
            )
        yield Static("Automatically delete older checkpoints to save space", classes="form-hint")
        yield Static("↓ Fewer = less disk  |  ↑ More = can go back further if needed", classes="impact-hint")
        
        # Resume from
        with Horizontal(classes="form-row"):
            yield Static("Resume From:", classes="form-label")
            with Horizontal(classes="form-input"):
                yield Input(
                    placeholder="(optional) path to checkpoint to resume",
                    id="input-resume-from",
                )
                yield Button("...", id="btn-browse-resume", variant="default")
        yield Static("Pick up where you left off from a previous run", classes="form-hint")
        
        yield Rule()
        yield Static("Audio Sampling", classes="section-header")
        
        # Sample every N epochs
        with Horizontal(classes="form-row"):
            yield Static("Sample Every N Epochs:", classes="form-label")
            yield Input(
                value="0",
                id="input-sample-every",
                validators=[Number(minimum=0, maximum=1000)],
                classes="form-input",
            )
        yield Static("Generate audio samples periodically to hear progress (0 = off)", classes="form-hint")
        yield Static("Useful for monitoring but adds time per epoch", classes="impact-hint")
    
    def on_mount(self) -> None:
        """
        Prepare and initialize UI inputs and validation when the screen is mounted.
        
        Populates form fields from saved preferences and an optional resume prefill, pre-fills target modules from the last estimation if available, updates the configuration preview, and starts periodic form validation.
        """
        # Auto-populate from preferences
        try:
            prefs = self.app.app_state.preferences
            ckpt = getattr(prefs, "default_checkpoint_dir", "")
            ds = getattr(prefs, "default_dataset_dir", "")
            out = getattr(prefs, "default_output_dir", "")
            if ckpt and not self.query_one("#input-checkpoint-dir", Input).value:
                self.query_one("#input-checkpoint-dir", Input).value = ckpt
            if ds and not self.query_one("#input-dataset-dir", Input).value:
                self.query_one("#input-dataset-dir", Input).value = ds
            if out and not self.query_one("#input-output-dir", Input).value:
                self.query_one("#input-output-dir", Input).value = out
        except Exception:
            pass

        # Apply resume pre-fill if supplied by RunHistoryScreen
        prefill = getattr(self, "_resume_prefill", None)
        if prefill:
            for key, widget_id in [
                ("output_dir", "#input-output-dir"),
                ("resume_from", "#input-resume-from"),
                ("checkpoint_dir", "#input-checkpoint-dir"),
                ("dataset_dir", "#input-dataset-dir"),
            ]:
                val = prefill.get(key, "")
                if val:
                    try:
                        self.query_one(widget_id, Input).value = val
                    except Exception:
                        pass

        # If estimation results exist, offer to pre-fill target modules
        try:
            modules = self.app.app_state.get_last_estimation_modules()
            if modules:
                mod_str = ",".join(modules[:16])
                try:
                    current = self.query_one("#input-target-modules", Input).value
                    if not current or current == "to_q,to_k,to_v,to_out.0":
                        self.query_one("#input-target-modules", Input).value = mod_str
                        self.notify(
                            f"Pre-filled {len(modules[:16])} modules from estimation",
                            timeout=5,
                        )
                except Exception:
                    pass
        except Exception:
            pass

        self._update_preview()

        # Set up validation
        self.set_interval(0.5, self._validate_form)
    
    def _validate_form(self) -> None:
        """
        Validate the training configuration form and update validation state.
        
        Checks that checkpoint, dataset, and output path fields are present; verifies checkpoint path exists; if the dataset path is a directory, verifies it contains at least one `.pt` file. Updates self._validation_errors with any found issues and updates the `#validation-status` widget to show the first error and count (in red) or a success message (in green).
        """
        errors = []
        
        # Check required paths
        checkpoint_dir = self.query_one("#input-checkpoint-dir", Input).value
        if not checkpoint_dir:
            errors.append("Checkpoint directory is required")
        elif not Path(checkpoint_dir).exists():
            errors.append(f"Checkpoint directory not found: {checkpoint_dir}")
        
        dataset_dir = self.query_one("#input-dataset-dir", Input).value
        if not dataset_dir:
            errors.append("Dataset directory is required")
        elif Path(dataset_dir).is_dir() and not list(Path(dataset_dir).glob("*.pt")):
            errors.append("Dataset directory has no .pt files -- preprocess first")
        
        output_dir = self.query_one("#input-output-dir", Input).value
        if not output_dir:
            errors.append("Output directory is required")
        
        # Update validation status
        self._validation_errors = errors
        status = self.query_one("#validation-status", Static)
        
        if errors:
            status.update(Text(f"⚠ {len(errors)} issue(s): {errors[0]}", style="red"))
        else:
            status.update(Text("✓ Configuration valid - ready to start", style="green"))
    
    def _update_preview(self) -> None:
        """
        Update the preview panel to reflect current form values.
        
        Builds a concise summary showing trainer type, epochs, effective batch (batch size × gradient accumulation), LoRA rank, and the model and dataset base names, and writes it to the preview content widget. Errors are silently ignored (useful during initial setup when widgets may be missing).
        """
        try:
            checkpoint = self.query_one("#input-checkpoint-dir", Input).value or "..."
            dataset = self.query_one("#input-dataset-dir", Input).value or "..."
            epochs = self.query_one("#input-epochs", Input).value or "100"
            batch_size = self.query_one("#input-batch-size", Input).value or "1"
            grad_accum = self.query_one("#input-grad-accum", Input).value or "4"
            rank = self.query_one("#input-rank", Input).value or "64"
            
            effective_batch = int(batch_size) * int(grad_accum)
            
            preview_text = (
                f"Trainer: {self.trainer_type} | "
                f"Epochs: {epochs} | "
                f"Effective batch: {effective_batch} ({batch_size}×{grad_accum}) | "
                f"LoRA rank: {rank}\n"
                f"Model: {Path(checkpoint).name if checkpoint != '...' else '...'} | "
                f"Dataset: {Path(dataset).name if dataset != '...' else '...'}"
            )
            
            self.query_one("#preview-content", Static).update(preview_text)
        except Exception:
            pass  # Ignore errors during initial setup
    
    def _has_widget(self, selector: str) -> bool:
        """
        Determine whether a widget matching the given selector exists in the screen.
        
        Parameters:
            selector (str): Selector string used to locate the widget.
        
        Returns:
            True if a widget matching `selector` is present, False otherwise.
        """
        try:
            self.query_one(selector)
            return True
        except Exception:
            return False

    @staticmethod
    def _safe_int(value: str, default: int) -> int:
        """
        Parse an integer from a string, falling back to the provided default on failure.
        
        Parameters:
            value (str): The string to parse as an integer. Empty or None will use the default.
            default (int): The value to return if parsing fails.
        
        Returns:
            int: The parsed integer, or `default` if `value` is empty or cannot be parsed.
        """
        try:
            return int(value) if value else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_float(value: str, default: float) -> float:
        """
        Parse a float from a string, falling back to a default value on invalid or empty input.
        
        Parameters:
            value (str): The string to parse as a float. If empty or not parseable, the function returns `default`.
            default (float): The value to return if `value` is empty or cannot be parsed as a float.
        
        Returns:
            float: The parsed float value, or `default` if parsing fails.
        """
        try:
            return float(value) if value else default
        except (ValueError, TypeError):
            return default

    def _get_config_dict(self) -> dict:
        """
        Constructs a training configuration dictionary from the current form inputs.
        
        The returned dictionary contains keys for required paths and variant, LoRA settings, training hyperparameters, advanced device/optimizer options, and logging/checkpointing values. It includes an optional "resume_from" entry when provided, a "cfg_ratio" key when this screen's trainer_type is "fixed", and expert-mode keys ("bias", "prefetch_factor", "persistent_workers", "log_heavy_every") when expert mode is enabled; when expert mode is disabled those expert fields are populated with sensible defaults.
        
        Returns:
            config (dict): Mapping of configuration keys to their resolved values:
                - Required: "checkpoint_dir", "dataset_dir", "output_dir", "variant"
                - LoRA: "rank", "alpha", "dropout", "attention_type", "target_modules"
                - Training: "epochs", "batch_size", "gradient_accumulation_steps", "learning_rate", "scheduler_type", "warmup_steps", "seed"
                - Advanced: "device", "precision", "weight_decay", "max_grad_norm", "num_workers", "pin_memory", "optimizer_type", "gradient_checkpointing", "offload_encoder"
                - Logging: "log_dir", "log_every_n_steps", "save_every_n_epochs", "keep_last_n", "sample_every_n_epochs"
                - Optional: "resume_from" (if provided)
                - Conditional: "cfg_ratio" (present when trainer_type == "fixed")
                - Expert: "bias", "prefetch_factor", "persistent_workers", "log_heavy_every" (present and populated from the form when expert mode is enabled; otherwise set to defaults)
        """
        config = {
            # Required
            "checkpoint_dir": self.query_one("#input-checkpoint-dir", Input).value,
            "dataset_dir": self.query_one("#input-dataset-dir", Input).value,
            "output_dir": self.query_one("#input-output-dir", Input).value,
            "variant": self._get_selected_variant(),
            
            # LoRA
            "rank": self._safe_int(self.query_one("#input-rank", Input).value, 64),
            "alpha": self._safe_int(self.query_one("#input-alpha", Input).value, 128),
            "dropout": self._safe_float(self.query_one("#input-dropout", Input).value, 0.0),
            "attention_type": self.query_one("#select-attention-type", Select).value,
            "target_modules": self.query_one("#input-target-modules", Input).value,
            
            # Training
            "epochs": self._safe_int(self.query_one("#input-epochs", Input).value, 100),
            "batch_size": self._safe_int(self.query_one("#input-batch-size", Input).value, 1),
            "gradient_accumulation_steps": self._safe_int(self.query_one("#input-grad-accum", Input).value, 4),
            "learning_rate": self._safe_float(self.query_one("#input-lr", Input).value, 1e-4),
            "scheduler_type": self.query_one("#select-scheduler", Select).value,
            "warmup_steps": self._safe_int(self.query_one("#input-warmup", Input).value, 500),
            "seed": self._safe_int(self.query_one("#input-seed", Input).value, 42),
            
            # Advanced
            "device": self.query_one("#select-device", Select).value,
            "precision": self.query_one("#select-precision", Select).value,
            "weight_decay": self._safe_float(self.query_one("#input-weight-decay", Input).value, 0.01),
            "max_grad_norm": self._safe_float(self.query_one("#input-max-grad-norm", Input).value, 1.0),
            "num_workers": self._safe_int(self.query_one("#input-num-workers", Input).value, 4),
            "pin_memory": self.query_one("#switch-pin-memory", Switch).value,
            "optimizer_type": self.query_one("#select-optimizer", Select).value if self._has_widget("#select-optimizer") else "adamw",
            "gradient_checkpointing": self.query_one("#switch-grad-ckpt", Switch).value if self._has_widget("#switch-grad-ckpt") else False,
            "offload_encoder": self.query_one("#switch-offload-encoder", Switch).value if self._has_widget("#switch-offload-encoder") else False,
            
            # Logging
            "log_dir": self.query_one("#input-log-dir", Input).value or "./logs",
            "log_every_n_steps": self._safe_int(self.query_one("#input-log-every", Input).value, 10),
            "save_every_n_epochs": self._safe_int(self.query_one("#input-save-every", Input).value, 10),
            "keep_last_n": self._safe_int(self.query_one("#input-keep-last", Input).value, 3),
            "sample_every_n_epochs": self._safe_int(self.query_one("#input-sample-every", Input).value, 0),
        }
        
        # Resume from (optional)
        resume_from = self.query_one("#input-resume-from", Input).value
        if resume_from:
            config["resume_from"] = resume_from
        
        # CFG ratio (fixed trainer only)
        if self.trainer_type == "fixed":
            try:
                config["cfg_ratio"] = float(self.query_one("#input-cfg-ratio", Input).value or 0.15)
            except Exception:
                config["cfg_ratio"] = 0.15
        
        # Expert mode settings
        if self._expert_mode:
            try:
                config["bias"] = self.query_one("#select-bias", Select).value
                config["prefetch_factor"] = self._safe_int(self.query_one("#input-prefetch-factor", Input).value, 2)
                config["persistent_workers"] = self.query_one("#switch-persistent-workers", Switch).value
                config["log_heavy_every"] = self._safe_int(self.query_one("#input-log-heavy-every", Input).value, 50)
            except Exception:
                pass  # Expert settings not available
        else:
            config["bias"] = "none"
            config["prefetch_factor"] = 2
            config["persistent_workers"] = True
            config["log_heavy_every"] = 50
        
        return config
    
    def _get_selected_variant(self) -> str:
        """
        Determine which model variant is currently selected in the UI.
        
        Returns:
            str: 'turbo', 'base', or 'sft' depending on the selected radio button; returns 'turbo' if no selection can be determined.
        """
        radio_set = self.query_one("#radio-variant", RadioSet)
        # Check which radio button is pressed
        try:
            if self.query_one("#variant-turbo", RadioButton).value:
                return "turbo"
            elif self.query_one("#variant-base", RadioButton).value:
                return "base"
            elif self.query_one("#variant-sft", RadioButton).value:
                return "sft"
        except Exception:
            pass
        return "turbo"
    
    # =========================================================================
    # Input Handlers
    # =========================================================================
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """
        Update the configuration preview when any input value changes.
        
        Parameters:
            event (Input.Changed): The input change event that triggered the update.
        """
        self._update_preview()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """
        Handle changes to Select widgets and update dependent UI state.
        
        If the changed select is the VRAM profile selector and the new value is not "auto", apply the corresponding VRAM profile presets. Always refresh the configuration preview after any select change.
        
        Parameters:
            event (Select.Changed): The select change event carrying the widget id and new value.
        """
        if event.select.id == "select-vram-profile" and event.value != "auto":
            self._apply_vram_profile(event.value)
        self._update_preview()

    _VRAM_PROFILES = {
        "comfortable": {"batch_size": "2", "grad_accum": "4", "rank": "64", "optimizer": "adamw", "grad_ckpt": False, "offload": False},
        "standard":    {"batch_size": "1", "grad_accum": "4", "rank": "64", "optimizer": "adamw", "grad_ckpt": False, "offload": False},
        "tight":       {"batch_size": "1", "grad_accum": "8", "rank": "32", "optimizer": "adamw8bit", "grad_ckpt": True, "offload": True},
        "minimal":     {"batch_size": "1", "grad_accum": "16", "rank": "16", "optimizer": "adamw8bit", "grad_ckpt": True, "offload": True},
    }

    def _apply_vram_profile(self, profile: str) -> None:
        """
        Apply a VRAM profile to populate related form fields and notify the user.
        
        If `profile` matches a key in the internal VRAM profiles map, updates the corresponding form fields (batch size, gradient accumulation, LoRA rank, optimizer, gradient checkpointing, and encoder offload) with the profile's presets and shows a brief notification.
        
        Parameters:
        	profile (str): Name of the VRAM profile to apply (must match a key in `_VRAM_PROFILES`).
        """
        preset = self._VRAM_PROFILES.get(profile)
        if not preset:
            return
        try:
            self.query_one("#input-batch-size", Input).value = preset["batch_size"]
            self.query_one("#input-grad-accum", Input).value = preset["grad_accum"]
            self.query_one("#input-rank", Input).value = preset["rank"]
            self.query_one("#select-optimizer", Select).value = preset["optimizer"]
            self.query_one("#switch-grad-ckpt", Switch).value = preset["grad_ckpt"]
            self.query_one("#switch-offload-encoder", Switch).value = preset["offload"]
        except Exception:
            pass
        self.notify(
            f"Applied '{profile}' VRAM profile -- you can override any value",
            timeout=3,
        )
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """
        Toggle the screen's expert mode when the expert-mode switch is changed.
        
        Parameters:
            event (Switch.Changed): The switch change event; if the switch id is "switch-expert-mode",
                its boolean `value` is applied to the expert mode state.
        """
        if event.switch.id == "switch-expert-mode":
            self._toggle_expert_mode(event.value)
    
    # =========================================================================
    # Button Handlers
    # =========================================================================
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Route button press events to their corresponding actions or file/dataset pickers.
        
        Parameters:
            event (Button.Pressed): The button press event. The handler inspects `event.button.id` and performs the matching action:
                - "btn-start": start training
                - "btn-cancel": cancel/close the screen
                - "btn-browse-checkpoint": open directory picker for checkpoint directory
                - "btn-browse-dataset": open dataset picker
                - "btn-browse-output": open directory picker for output directory
                - "btn-browse-logs": open directory picker for log directory
                - "btn-browse-resume": open directory picker for resume checkpoint
        """
        button_id = event.button.id
        
        if button_id == "btn-start":
            self.action_start_training()
        elif button_id == "btn-cancel":
            self.action_cancel()
        elif button_id == "btn-browse-checkpoint":
            self._open_file_picker("Select Checkpoint Directory", "#input-checkpoint-dir", select_directory=True)
        elif button_id == "btn-browse-dataset":
            self._open_dataset_picker()
        elif button_id == "btn-browse-output":
            self._open_file_picker("Select Output Directory", "#input-output-dir", select_directory=True)
        elif button_id == "btn-browse-logs":
            self._open_file_picker("Select Log Directory", "#input-log-dir", select_directory=True)
        elif button_id == "btn-browse-resume":
            self._open_file_picker("Select Checkpoint to Resume", "#input-resume-from", select_directory=True)
    
    def _open_dataset_picker(self) -> None:
        """
        Open a dataset picker modal and update the dataset directory input with the user's selection.
        
        If the dataset input currently contains a valid path, that path is used as the modal's start directory; otherwise the current working directory is used. When the user selects a path, the function sets the value of the input widget with id `#input-dataset-dir`.
        """
        from acestep.training_v2.tui.screens.dataset_browser import DatasetPickerModal

        current_val = ""
        try:
            current_val = self.query_one("#input-dataset-dir", Input).value
        except Exception:
            pass
        start = Path(current_val) if current_val and Path(current_val).exists() else Path.cwd()

        def _handle_result(path: str | None) -> None:
            """
            Set the dataset directory input widget to the given path if provided.
            
            Parameters:
                path (str | None): Filesystem path to set in the dataset directory input. If `None`, no action is taken.
            
            Notes:
                Any errors encountered while locating or updating the input widget are ignored.
            """
            if path is not None:
                try:
                    self.query_one("#input-dataset-dir", Input).value = path
                except Exception:
                    pass

        self.app.push_screen(DatasetPickerModal(start_path=start), _handle_result)

    def _open_file_picker(self, title: str, target_input_id: str, select_directory: bool = True) -> None:
        """
        Open a file/directory picker modal and fill the specified input with the selected path.
        
        Parameters:
            title (str): Window title shown in the picker modal.
            target_input_id (str): Selector or widget id for the Input whose value will be set to the chosen path.
            select_directory (bool): If True, the picker will select directories; if False, it will select files.
        
        Behavior:
            Presents a file picker modal to the user. When the user selects a path, the function updates the target Input's value with the selected path string. If the target input cannot be found or the selection is cancelled, no change is made.
        """
        from acestep.training_v2.tui.widgets.file_picker import FilePickerModal
        from pathlib import Path

        current_val = ""
        try:
            current_val = self.query_one(target_input_id, Input).value
        except Exception:
            pass
        start = Path(current_val) if current_val and Path(current_val).exists() else Path.cwd()

        async def _handle_result(path) -> None:
            """
            Set the target input widget's value from a path result.
            
            If `path` is not None, attempts to assign `str(path)` to the Input widget referenced by `target_input_id`. Any errors during widget lookup or assignment are suppressed.
            
            Parameters:
                path (str | Path | None): The selected file or directory path to apply to the target input; if None, no change is made.
            """
            if path is not None:
                try:
                    self.query_one(target_input_id, Input).value = str(path)
                except Exception:
                    pass

        self.app.push_screen(
            FilePickerModal(
                title=title,
                start_path=start,
                select_directory=select_directory,
            ),
            _handle_result,
        )

    def _toggle_expert_mode(self, enabled: bool) -> None:
        """
        Enable or disable expert-mode UI and internal expert flag.
        
        When enabled, reveals the expert-settings container and sets the internal expert-mode flag; when disabled, hides the expert-settings container. Enabling also displays a transient warning notification about potential training stability impacts.
        
        Parameters:
            enabled (bool): True to enable and reveal expert settings, False to disable and hide them.
        """
        self._expert_mode = enabled
        try:
            expert_section = self.query_one("#expert-settings", Container)
            if enabled:
                expert_section.remove_class("hidden")
                self.notify(
                    "⚠️ Expert mode enabled - these settings can affect training stability",
                    severity="warning",
                    timeout=5,
                )
            else:
                expert_section.add_class("hidden")
        except Exception:
            pass
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    def action_start_training(self) -> None:
        """
        Initiates a training run from the current form if validation passes.
        
        Validates the form; if validation fails, shows an error notification and aborts. If validation succeeds, constructs the training configuration and a RunInfo object, persists last-used paths to app state (errors ignored), and opens the TrainingMonitorScreen with the run and config.
        """
        self._validate_form()
        
        if self._validation_errors:
            self.notify(
                f"Cannot start: {self._validation_errors[0]}",
                severity="error",
                timeout=5,
            )
            return
        
        # Build config and start training
        config = self._get_config_dict()
        
        # Create run info
        from acestep.training_v2.tui.state import RunInfo
        
        run_name = Path(config["output_dir"]).name
        run = RunInfo(
            name=run_name,
            trainer_type=self.trainer_type,
            status="pending",
            total_epochs=config["epochs"],
            output_dir=config["output_dir"],
            checkpoint_dir=config["checkpoint_dir"],
            dataset_dir=config["dataset_dir"],
        )
        
        # Persist last-used paths for next time
        try:
            self.app.app_state.save_last_paths(
                checkpoint_dir=config.get("checkpoint_dir", ""),
                dataset_dir=config.get("dataset_dir", ""),
                output_dir=config.get("output_dir", ""),
            )
        except Exception:
            pass

        # Push to training monitor screen
        from acestep.training_v2.tui.screens.training_monitor import TrainingMonitorScreen

        self.app.push_screen(TrainingMonitorScreen(run=run, config=config))
    
    def action_cancel(self) -> None:
        """Cancel and go back."""
        self.app.pop_screen()
    
    def action_reset(self) -> None:
        """
        Reset the configuration form to its default values.
        
        Clears checkpoint, dataset, and output path fields; clears the resume-from field; restores default LoRA and training values (rank 64, alpha 128, epochs 100, seed 42); updates the preview panel and shows a brief notification indicating the reset.
        """
        self.query_one("#input-checkpoint-dir", Input).value = ""
        self.query_one("#input-dataset-dir", Input).value = ""
        self.query_one("#input-output-dir", Input).value = ""
        self.query_one("#input-rank", Input).value = "64"
        self.query_one("#input-alpha", Input).value = "128"
        self.query_one("#input-epochs", Input).value = "100"
        self.query_one("#input-seed", Input).value = "42"
        self.query_one("#input-resume-from", Input).value = ""
        self._update_preview()
        self.notify("Form reset to defaults", timeout=2)
    
    def action_toggle_expert(self) -> None:
        """
        Toggle the screen's expert mode state and update the expert-mode switch widget.
        
        Flips the expert-mode UI switch to the opposite state, causing the screen to enter or exit expert mode.
        """
        try:
            switch = self.query_one("#switch-expert-mode", Switch)
            switch.value = not switch.value
        except Exception:
            pass
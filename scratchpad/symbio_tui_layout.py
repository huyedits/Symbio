"""HIG-informed Symbio TUI layout (Textual).

Principles applied:
- Clear hierarchy: persistent title bar, primary nav rail, content region, status footer.
- 8-pt grid for outer padding/gaps, 4-pt for inner element spacing.
- Descriptive feedback states: idle/working/done/failed with semantic color + label change.
- Deliberate chrome: single title bar, one divider, lightweight cards, no decorative borders everywhere.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, ProgressBar, DataTable, Button, Label
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text

CSS = """
Screen {
    background: #0e1412;
    color: #efe7da;
}
Header, Footer {
    background: #141c19;
    color: #efe7da;
}
#body {
    height: 1fr;
}
 #rail {
    width: 24;
    background: #141c19;
    padding: 1 1;
    border-right: heavy #2b3f39;
}
#content {
    padding: 1 2;
}
.card {
    background: #1a2420;
    border: round #2b3f39;
    padding: 1 2;
    margin-bottom: 1;
}
.state-idle { color: #a39684; }
.state-working { color: #58b8aa; }
.state-done { color: #4aa06a; }
.state-failed { color: #b3361f; }
#progress {
    margin-top: 1;
}
"""

NAV = [("Chat", "chat"), ("Skills", "skills"), ("Activity", "activity"), ("Settings", "settings")]

class SymbioTUI(App):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1-4", "nav", "Section", show=False),
        Binding("r", "retry", "Retry"),
    ]
    CSS = CSS

    state = reactive("idle")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, name="Symbio")
        with Container(id="body"):
            with Horizontal():
                with Vertical(id="rail"):
                    for idx, (label, key) in enumerate(NAV, 1):
                        yield Button(f"{idx} {label}", id=f"nav-{key}", variant="primary")
                with Vertical(id="content"):
                    yield Static("Primary workspace", classes="card", id="title-card")
                    yield DataTable(id="table")
                    yield ProgressBar(id="progress", show_eta=False, show_percentage=True)
                    yield Static("State: idle", id="state")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        table.add_columns("Item", "Status", "Detail")
        table.add_row("Adapter bake", "working", "iter 42 / 120")
        table.add_row("Golden regression", "passed", "13/13")
        table.add_row("RAG sync", "done", "2 min ago")
        self.state = "idle"

    def watch_state(self, value: str) -> None:
        label = self.query_one("#state", Static)
        label.update(f"State: {value}")
        label.set_classes(f"state-{value}")

    def action_retry(self) -> None:
        self.state = "working"
        self.query_one("#progress", ProgressBar).update(total=100, progress=35)


if __name__ == "__main__":
    SymbioTUI().run()

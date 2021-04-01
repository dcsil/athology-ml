import typer

import athology_ml.ml.jump_detection.main as jump_detection_cli

app = typer.Typer(help="A simple CLI interface for the athology-ml repository.")
app.add_typer(jump_detection_cli.app, name="jump-detection")

if __name__ == "__main__":
    app()

import typer

from athology_ml.ml.jump_detection import train

app = typer.Typer()
app.add_typer(train.app, name="jump-detection")

if __name__ == "__main__":
    app()

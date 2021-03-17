import kerastuner
import typer
from kerastuner.tuners import RandomSearch
from athology_ml.ml.jump_detection import preprocessing
from athology_ml.ml.jump_detection.model import JumpDetector
from athology_ml.ml.jump_detection.util import KERAS_TUNER_SEED, print_baselines, set_seeds

app = typer.Typer(callback=set_seeds)


@app.command()
def train(
    directory: str = typer.Argument(
        ...,
        help=(
            "A directory that contains three CSVs: train.tsv, valid.tsv and test.tsv that will be"
            " used to train, tune and evaluate the model."
        ),
    ),
    batch_size: int = typer.Option(16, help="Batch size to use during tuning."),
    num_timesteps: int = typer.Option(128, help="Number of timesteps to include in each example."),
) -> None:
    """Train a model to perform jump detection on raw accelerometer data."""
    train_dataset, valid_dataset, test_dataset = preprocessing.get_datasets(
        directory, batch_size, num_timesteps
    )
    features, labels = preprocessing.get_features_and_labels(train_dataset)
    classifier_bias_init = preprocessing.get_classifier_bias_init(labels)
    normalizer = preprocessing.get_normalizer(features)

    typer.secho(
        f"Based on the training dataset, classifiers bias will be initialized to {classifier_bias_init}"
    )

    # Print baselines to console
    print_baselines(labels)

    hypermodel = JumpDetector(normalizer=normalizer, classifier_bias_init=classifier_bias_init)

    tuner = RandomSearch(
        hypermodel,
        objective=kerastuner.Objective("val_precision", direction="max"),
        max_trials=30,
        seed=KERAS_TUNER_SEED,
        allow_new_entries=True,
        directory="tuning",
        project_name=f"jump_detection_{batch_size}_{num_timesteps}",
    )
    tuner.search_space_summary()

    tuner.search(
        train_dataset,
        epochs=10,
        # callbacks=callbacks,
        validation_data=valid_dataset,
    )


if __name__ == "__main__":
    app()

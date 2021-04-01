from typing import Union
from pathlib import Path

import kerastuner
import typer
from athology_ml import msg
from athology_ml.ml.jump_detection import preprocessing
from athology_ml.ml.jump_detection.model import JumpDetector
from athology_ml.ml.jump_detection.util import KERAS_TUNER_SEED, print_baselines, set_seeds
from kerastuner.tuners import RandomSearch
from tensorflow.keras import mixed_precision

app = typer.Typer(
    callback=set_seeds,
    help="A simple CLI for training the jump detection model.",
)


@app.command()
def tune(
    dataset_dir: str = typer.Argument(
        ...,
        help=(
            "A directory that contains three subdirectories: train, valid and test, each containing"
            " the CSV files that will be used to train, tune and evaluate the model."
        ),
    ),
    output_dir: Union[str, Path] = typer.Argument(
        ...,
        help=(
            "Path to a directory where the results of the hyperparameter tuning will be saved,"
            " including the resulting best model found during the search."
        ),
    ),
    batch_size: int = typer.Option(16, help="Batch size to use during tuning."),
    num_timesteps: int = typer.Option(128, help="Number of timesteps to include in each example."),
    num_epochs: int = typer.Option(20, help="Number of epochs to train for during each trial."),
    max_trials: int = typer.Option(
        25, help="The maximum number of trials to run when searching for optimal hyperparameters."
    ),
    executions_per_trial: int = typer.Option(
        1,
        help=(
            "The number of times to repeat each trial. The bigger the number, the longer"
            " tuning will take but the less variable the results will be."
        ),
    ),
    use_mixed_precision: bool = typer.Option(
        False, help="Whether or not mixed precision should be used during training."
    ),
) -> None:
    """Trains a jump detection model from scratch, tuning the hyperparameters using KerasTuner."""
    msg.divider("Preprocessing")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_name = f"jump_detection_{batch_size}_{num_timesteps}"

    train_dataset, valid_dataset, test_dataset = preprocessing.get_datasets(
        dataset_dir, batch_size, num_timesteps
    )

    msg.good(f"Loaded training data at {dataset_dir}")

    features, labels = preprocessing.get_features_and_labels(train_dataset)
    classifier_bias_init = preprocessing.get_classifier_bias_init(labels)
    normalizer = preprocessing.get_normalizer(features)

    msg.info(
        f"Based on the training dataset, classifiers bias will be initialized to {classifier_bias_init:.4f}"
    )

    msg.divider("Baselines")
    print_baselines(labels)

    msg.divider("Tuning")
    if use_mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    hypermodel = JumpDetector(normalizer=normalizer, classifier_bias_init=classifier_bias_init)

    tuner = RandomSearch(
        hypermodel=hypermodel,
        objective=kerastuner.Objective("val_loss", direction="min"),
        max_trials=max_trials,
        seed=KERAS_TUNER_SEED,
        tune_new_entries=True,
        allow_new_entries=True,
        executions_per_trial=executions_per_trial,
        directory=str(output_dir),
        project_name=project_name,
    )
    tuner.search_space_summary()

    tuner.search(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
    )

    msg.divider("Saving")
    model = tuner.get_best_models(num_models=1)[-1]
    best_model_fp = str(output_dir / project_name / "best_model.tf")
    model.save(best_model_fp)
    msg.good(f"Saved the best model to: {best_model_fp}")


if __name__ == "__main__":
    app()

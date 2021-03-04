import kerastuner
import typer
from kerastuner.tuners import RandomSearch
from ml.jump_detection import preprocessing
from ml.jump_detection.model import JumpPredictor
from ml.jump_detection.util import KERAS_TUNER_SEED, print_baselines, set_seeds

app = typer.Typer(callback=set_seeds)

BUFFER_SIZE = 10000


@app.command()
def main(directory: str) -> None:
    train_dataset, valid_dataset, test_dataset = preprocessing.get_datasets(directory)
    features, labels = preprocessing.get_features_and_labels(train_dataset)
    classifier_bias_init = preprocessing.get_classifier_bias_init(labels)
    normalizer = preprocessing.get_normalizer(features)

    typer.secho(
        f"Based on the training dataset, classifiers bias will be initialized to {classifier_bias_init}"
    )

    # Print baselines to console
    print_baselines(labels)

    # Set the generator up to shuffle data on each epoch.
    # See https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    train_dataset = train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)

    hypermodel = JumpPredictor(normalizer=normalizer, classifier_bias_init=classifier_bias_init)

    class_weight_0 = 0.5
    class_weight_1 = 5

    tuner = RandomSearch(
        hypermodel,
        objective=kerastuner.Objective("val_recall", direction="max"),
        max_trials=10,
        seed=KERAS_TUNER_SEED,
        allow_new_entries=True,
        directory="tuning",
        project_name=f"{class_weight_0}-{class_weight_1}",
    )
    tuner.search_space_summary()

    tuner.search(
        train_dataset,
        epochs=2,
        # callbacks=callbacks,
        validation_data=valid_dataset,
        class_weight={0: class_weight_0, 1: class_weight_1},
    )


if __name__ == "__main__":
    app()

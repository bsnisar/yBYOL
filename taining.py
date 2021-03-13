import logging
import sys
import click
import os
import pandas
import os.path

from parastash import datasets, models, models_hub
from parastash import transforms

import metrics

# noinspection PyArgumentList
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


@click.command()
@click.option('--base_model', default='instagram', help='The Teacher model')
@click.option('--input_shape', default=244, help='Model input shape side length')
@click.option('--output_dimension', default=128, help='The dimension of the output embeddings')
@click.option('--freeze_base_network', default=False, is_flag=True, help='Freeze base network weights.')
@click.option('--train_dir', help='Training dataset', required=True)
@click.option('--test_dir', help='Testing dataset', required=True)
@click.option('--metrics_df', help='metrics validation csv', required=True)
@click.option('--batch_size', default=64, help='input batch size for training.')
@click.option('--test_batch_size', default=64, help='input batch size for testing.')
@click.option('--epochs', default=10, help='number of epochs to train.')
@click.option('--save_model_dir', default=f"{os.environ['HOME']}/_models", help='Output of the model.')
def cli(base_model, input_shape, output_dimension,
        freeze_base_network, train_dir, test_dir, metrics_df, batch_size, test_batch_size, epochs, save_model_dir):

    net = models_hub.ExternalModel(name=base_model)

    model = models.BYOL(
        external_net=net,
        input_shape=input_shape,
        output_dimension=output_dimension,
        freeze_base_network=freeze_base_network,
    )

    logger.info(f"Initialize self-learning framework from '{base_model}' net: train_dir={train_dir},"
                f" test_dir={test_dir} epochs={epochs} batch_size={batch_size} model_dir={save_model_dir}")

    train_transformations = transforms.byol_augmentations()
    train_dataset = datasets.FolderDataset(
        train_dir, transformations=train_transformations
    )

    test_transformations = transforms.basic_augmentations()
    test_dataset = datasets.FolderDataset(
        test_dir, transformations=test_transformations
    )

    logger.info(f"Loaded dataset of {len(train_dataset)} images")
    logger.info(f"Loaded test dataset of {len(test_dataset)} images")
    logger.info("Training model:")
    loader_workers = min(os.cpu_count() - 2, 0)
    train_loader = train_dataset.create_loader(
        batch_size, shuffle=True, drop_last=True, num_workers=loader_workers
    )
    model.train(epochs=epochs, loader=train_loader)
    logger.info("Computing test embeddings")

    test_loader = test_dataset.create_loader(
        batch_size=test_batch_size, shuffle=False, num_workers=loader_workers
    )

    logger.info("Computing metrics:")
    test_embeddings = model.predict_from_loader(test_loader)
    stats = metrics.labeled_collections_precision_recall_at_k(
        test_embeddings, test_loader, pandas.read_csv(metrics_df)
    )
    for k,counts in stats.items():
        logger.info(f"   [top{k}] {counts}")
    logger.info("Saving model:")
    model.save(save_model_dir)


if __name__ == '__main__':
    cli()

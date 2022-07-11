import os
import pprint

import torch

from tqdm import tqdm
from time import sleep

from pypc import utils
from pypc import datasets
from pypc import optim
from pypc.models import PCModel


def main(cf):
    print(f"\nStarting supervised experiment {cf.logdir}: --seed {cf.seed} --device {utils.DEVICE}")
    pprint.pprint(cf)
    os.makedirs(cf.logdir, exist_ok=True)
    utils.seed(cf.seed)
    utils.save_json({k: str(v) for (k, v) in cf.items()}, cf.logdir + "config.json")

    train_dataset = datasets.MNIST(
        train=True, scale=cf.label_scale, size=cf.train_size, normalize=cf.normalize
    )
    test_dataset = datasets.MNIST(
        train=False, scale=cf.label_scale, size=cf.test_size, normalize=cf.normalize
    )
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.batch_size)
    print(f"Loaded data [train batches: {len(train_loader)} test batches: {len(test_loader)}]")

    model = PCModel(
        nodes=cf.nodes,
        mu_dt=cf.mu_dt,
        act_fn=cf.act_fn,
        use_bias=cf.use_bias,
        kaiming_init=cf.kaiming_init,
    )
    optimizer = optim.get_optim(
        model.params,
        cf.optim,
        cf.lr,
        batch_scale=cf.batch_scale,
        grad_clip=cf.grad_clip,
        weight_decay=cf.weight_decay,
    )

    with torch.no_grad():
        metrics = {"acc": []}
        for epoch in range(1, cf.n_epochs + 1):

            print(f"\nTrain @ epoch {epoch} ({len(train_loader)} batches)")
            sleep(0.1)
            for batch_id, (img_batch, label_batch) in enumerate(tqdm(train_loader)):
                model.train_batch_supervised(
                    img_batch, label_batch, cf.n_train_iters, fixed_preds=cf.fixed_preds_train
                )
                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )

            if epoch % cf.test_every == 0:
                acc = 0
                for _, (img_batch, label_batch) in enumerate(tqdm(test_loader)):
                    label_preds = model.test_batch_supervised(img_batch)
                    acc += datasets.accuracy(label_preds, label_batch)
                metrics["acc"].append(acc / len(test_loader))
                print("\nTest @ epoch {} / Accuracy: {:.4f}".format(epoch, acc / len(test_loader)))

            utils.save_json(metrics, cf.logdir + "metrics.json")


if __name__ == "__main__":
    cf = utils.AttrDict()
    cf.seeds = [0]

    for seed in cf.seeds:

        # experiment params
        cf.seed = seed
        cf.n_epochs = 30
        cf.test_every = 1
        cf.log_every = 100
        cf.logdir = f"data/supervised/{seed}/"

        # dataset params
        cf.train_size = None
        cf.test_size = None
        cf.label_scale = None
        cf.normalize = False

        # optim params
        cf.optim = "Adam"
        cf.lr = 1e-4
        cf.batch_size = 64
        cf.batch_scale = False
        cf.grad_clip = 50
        cf.weight_decay = None

        # inference params
        cf.mu_dt = 0.01
        cf.n_train_iters = 50
        cf.fixed_preds_train = True

        # model params
        cf.use_bias = True
        cf.kaiming_init = False
        cf.nodes = [784, 300, 100, 10]
        cf.act_fn = utils.ReLU()

        main(cf)

import os
import pprint

import torch

from pypc import utils
from pypc import datasets
from pypc import optim
from pypc.models import PCModel


def main(cf):
    print(f"\nStarting generative experiment {cf.logdir}: --seed {cf.seed} --device {utils.DEVICE}")
    pprint.pprint(cf)
    os.makedirs(cf.logdir, exist_ok=True)
    os.makedirs(cf.imgdir, exist_ok=True)
    utils.seed(cf.seed)
    utils.save_json({k: str(v) for (k, v) in cf.items()}, cf.logdir + "config.json")

    train_dataset = datasets.MNIST(train=True, scale=cf.label_scale, size=cf.train_size, normalize=cf.normalize)
    test_dataset = datasets.MNIST(train=False, scale=cf.label_scale, size=cf.test_size, normalize=cf.normalize)
    train_loader = datasets.get_dataloader(train_dataset, cf.batch_size)
    test_loader = datasets.get_dataloader(test_dataset, cf.batch_size)
    print(f"Loaded data [train batches: {len(train_loader)} test batches: {len(test_loader)}]")

    model = PCModel(
        nodes=cf.nodes, mu_dt=cf.mu_dt, act_fn=cf.act_fn, use_bias=cf.use_bias, kaiming_init=cf.kaiming_init
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
            for batch_id, (img_batch, label_batch) in enumerate(train_loader):
                model.train_batch_generative(
                    img_batch, label_batch, cf.n_train_iters, fixed_preds=cf.fixed_preds_train
                )
                optimizer.step(
                    curr_epoch=epoch,
                    curr_batch=batch_id,
                    n_batches=len(train_loader),
                    batch_size=img_batch.size(0),
                )

            if epoch % cf.test_every == 0:
                print(f"\nTest @ epoch {epoch}")
                acc = 0
                for _, (img_batch, label_batch) in enumerate(test_loader):
                    label_preds = model.test_batch_generative(
                        img_batch, cf.n_test_iters, init_std=cf.init_std, fixed_preds=cf.fixed_preds_test
                    )
                    acc += datasets.accuracy(label_preds, label_batch)
                metrics["acc"].append(acc / len(test_loader))
                print("Accuracy: {:.4f}".format(acc / len(test_loader)))

                _, label_batch = next(iter(test_loader))
                img_preds = model.forward(label_batch)
                datasets.plot_imgs(img_preds, cf.imgdir + f"{epoch}.png")

            utils.save_json(metrics, cf.logdir + "metrics.json")


if __name__ == "__main__":
    cf = utils.AttrDict()
    cf.seeds = [0]

    for seed in cf.seeds:

        # experiment params
        cf.seed = seed
        cf.n_epochs = 20
        cf.test_every = 1
        cf.logdir = f"data/generative/{seed}/"
        cf.imgdir = cf.logdir + "imgs/"

        # dataset params
        cf.train_size = None
        cf.test_size = None
        cf.label_scale = None
        cf.normalize = True

        # optim params
        cf.optim = "Adam"
        cf.lr = 1e-4
        cf.batch_size = 64
        cf.batch_scale = True
        cf.grad_clip = None
        cf.weight_decay = None

        # inference params
        cf.mu_dt = 0.01
        cf.n_train_iters = 50
        cf.n_test_iters = 200
        cf.init_std = 0.01
        cf.fixed_preds_train = False
        cf.fixed_preds_test = False

        # model params
        cf.use_bias = True
        cf.kaiming_init = False
        cf.nodes = [10, 100, 300, 784]
        cf.act_fn = utils.Tanh()

        main(cf)

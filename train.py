""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models import SearchCNNController
from architect import Architect
# from visualize import plot_nor,plot_red
import h5py
import torch.utils.data as Data

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(
    config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def shuffle_data(arr1, arr2):
    per = np.random.permutation(arr1.shape[0])
    shuffle_arr1 = arr1[per, :, :]
    shuffle_arr2 = arr2[per, :, :]
    return shuffle_arr1, shuffle_arr2


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])
    # torch.cuda.set_device(2)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    x_path = '/media/omnisky/Tiger/JY_GD/MLF-ICS_code/data/patch_64_200x1024.mat'
    inputs = h5py.File(x_path)["patch_input"]
    target = h5py.File(x_path)["patch_target"]
    print(inputs.shape)
    target = np.transpose(target, [0, 2, 1])
    print(inputs.shape)
    inputs = np.transpose(inputs, [0, 2, 1])
    print(inputs.shape)
    val_inputs = inputs[102400:204800, :, :]
    val_target = target[102400:204800, :, :]

    val_inputs_shuffle, val_target_shuffle = shuffle_data(
        val_inputs, val_target)
    inputs = inputs[0:102400, :, :]
    target = target[0:102400, :, :]
    inputs_shuffle, target_shuffle = shuffle_data(inputs, target)
    print(inputs.shape)

    val_inputs_s = np.expand_dims(val_inputs_shuffle, 1)
    train_inputs = np.expand_dims(inputs_shuffle, 1)
    val_target_s = np.expand_dims(val_target_shuffle, 1)
    train_target = np.expand_dims(target_shuffle, 1)
    val_inputs_s = val_inputs_s[0:102400:10, :, :, :]
    val_target_s = val_target_s[0:102400:10, :, :, :]
    train_inputs = train_inputs[0:102400:10, :, :, :]
    train_target = train_target[0:102400:10, :, :, :]

    test_path = "/media/omnisky/Tiger/JY_GD/MLF-ICS_code/data/test_512x512withlaplace.mat"
    test_inputs = h5py.File(test_path)["inputs"]
    test_target = h5py.File(test_path)["targets"]
    test_inputs = np.transpose(test_inputs, [0, 2, 1])
    test_target = np.transpose(test_target, [0, 2, 1])
    test_inputs = test_inputs[0:500, :, :]
    test_target = test_target[0:500, :, :]

    test_inputs = np.expand_dims(test_inputs, 1)
    test_target = np.expand_dims(test_target, 1)

    # get data with meta info
    # input_size, input_channels, n_classes, train_data = utils.get_data(
    #     config.dataset, config.data_path, cutout_length=0, validation=False)
    input_channels = 1

    net_crit = nn.MSELoss().to(device)
    model = SearchCNNController(input_channels, config.init_channels, config.layers,
                                net_crit, device_ids=config.gpus)
    model = model.to(device)
    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)

    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    torch_dataset_val = Data.TensorDataset(torch.from_numpy(
        val_inputs_s), torch.from_numpy(val_target_s))
    valid_loader = Data.DataLoader(
        dataset=torch_dataset_val,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.workers
    )

    torch_dataset = Data.TensorDataset(torch.from_numpy(
        train_inputs), torch.from_numpy(train_target))
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.workers
    )

    torch_dataset_test = Data.TensorDataset(
        torch.from_numpy(test_inputs), torch.from_numpy(test_target))
    test_loader = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.workers
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    for epoch in range(config.epochs):
        print(epoch)
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(train_loader, valid_loader, model,
              architect, w_optim, alpha_optim, lr, epoch)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        validate(test_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        # plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        # caption = "Epoch {}".format(epoch+1)
        # plot_nor(genotype.normal, plot_path + "-normal", caption)
        # plot_red(genotype.reduce, plot_path + "-reduce", caption)

        # save
        utils.save_checkpoint(model, config.path)
        print("")


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X = trn_X.type(torch.FloatTensor)
        trn_y = trn_y.type(torch.FloatTensor)
        val_X = val_X.type(torch.FloatTensor)
        val_y = val_y.type(torch.FloatTensor)
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(
            device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(
            device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        if epoch >= 10:
            alpha_optim.zero_grad()
            architect.unrolled_backward(
                trn_X, trn_y, val_X, val_y, lr, w_optim)
            alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        loss = model.loss(trn_X, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()
        losses.update(loss.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.10f} ".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses))
        cur_step += 1

    logger.info(
        "Train: [{:2d}/{}] Final loss {:.10f}".format(epoch+1, config.epochs, losses.avg))


def validate(test_loader, model, epoch, cur_step):
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(test_loader):
            X = X.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True)
            N = X.size(0)

            loss = model.loss_val(X, y)
            losses.update(loss.item(), N)

            if step % config.print_freq == 0 or step == len(test_loader)-1:
                logger.info(
                    "test: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.10f} ".format(
                        epoch+1, config.epochs, step, len(test_loader)-1, losses=losses))

    logger.info(
        "test: [{:2d}/{}] Final loss {:.10f}".format(epoch+1, config.epochs, losses.avg))

    return losses.avg


if __name__ == "__main__":
    main()

import os
import time

import torch

from argumentation.cutmix import cutmix_data, cutmix_criterion
from argumentation.mixup import mixup_data, mixup_criterion


def train(model, data_loader, device, optimizer, loss_function, epoch, batch_size, warmup_dict=None, writer=None,
          argumentation=None, alpha=0.2):
    """
    :param model: cnn model
    :param data_loader: the data loader
    :param device: cpu or gpu
    :param optimizer: the optimizer
    :param loss_function: the loss function
    :param epoch: the number of iteration
    :param batch_size: the batch size
    :param warmup_dict: whether to warmup or not
    :param writer: whether to writer logs or not
    :param argumentation: the type of argumentation: mixup, cutout, cutmix
    :param alpha: the parameter of beta(\alpha, \alpha), default=0.2
    :return:
    """
    loss_list = []
    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(data_loader):
        if device != "cpu":
            labels = labels.to(device)
            images = images.to(device)
        # 数据增强: mixup 与cutmix (cutout在数据处理时已经进行了处理)
        if argumentation == "mixup":
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
        elif argumentation == "cutmix":
            images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = cutmix_criterion(loss_function, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        n_iter = (epoch - 1) * len(data_loader) + batch_index + 1

        last_layer = list(model.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * batch_size + len(images),
            total_samples=len(data_loader.dataset)
        ))
        # save training loss
        loss_list.append(loss.item())

        if writer is not None:

            # update training loss for each iteration
            writer.add_scalar('Train/loss', loss.item(), n_iter)

        if warmup_dict is not None:
            warmup_scheduler = warmup_dict["warmup_scheduler"]
            if epoch <= warmup_dict["warmup_num"]:
                warmup_scheduler.step()

    if writer is not None:
        for name, param in model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    return(sum(loss_list))


@torch.no_grad()
def eval_training(model, data_loader, device, loss_function, epoch=0, writer=None):
    start = time.time()
    model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in data_loader:

        if device != "cpu":
            labels = labels.to(device)
            images = images.to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if device != "cpu":
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(data_loader.dataset),
        correct.float() / len(data_loader.dataset),
        finish - start
    ))
    print()

    # add information to tensorboard
    if writer is not None:
        writer.add_scalar('Test/Average loss', test_loss / len(data_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(data_loader.dataset), epoch)

    return correct.float() / len(data_loader.dataset), test_loss / len(data_loader.dataset)
import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model, yolo_loss, loss_history, eval_callback, optimizer,
                  epoch, epoch_step, loader, val_loader, total_epochs, save_dir):
    loss = 0
    device = next(model.parameters()).device
    model.train()
    progress = tqdm(total=epoch_step, desc='Epoch {}/{}'.format(epoch + 1, total_epochs),
                    mininterval=0.3)

    for iteration, (images, targets) in enumerate(loader):
        if iteration >= epoch_step:
            break
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        optimizer.zero_grad()
        outputs = model(images)
        batch_loss = sum(yolo_loss(index, output, targets)
                         for index, output in enumerate(outputs))
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        progress.set_postfix(loss=loss / (iteration + 1), lr=get_lr(optimizer))
        progress.update(1)

    progress.close()
    model.eval()
    train_loss = loss / epoch_step
    val_loss = 0
    val_images = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            outputs = model(images)
            batch_loss = sum(yolo_loss(index, output, targets)
                             for index, output in enumerate(outputs))
            val_loss += batch_loss.item() * len(images)
            val_images += len(images)
    val_loss /= val_images
    loss_history.append_loss(epoch + 1, train_loss, val_loss)
    val_metrics = eval_callback.on_epoch_end(epoch + 1, model)
    print('Epoch {}/{} - train loss: {:.3f}, validation loss: {:.3f}'.format(
        epoch + 1, total_epochs, train_loss, val_loss))

    if val_metrics is not None and eval_callback.best_epoch == epoch + 1:
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_epoch_weights.pth'))

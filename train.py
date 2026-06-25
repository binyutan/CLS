from setting import model_map, dataset_map, parse_opts

import os
import time
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler


def save_checkpoint(model, epoch, args, best_acc=0, optimizer=None, scheduler=None):
    """
    Saves the model checkpoint to disk.

    Args:
        model (torch.nn.Module): The model to be saved.
        epoch (int): Current epoch index.
        args (Namespace): Argument parser containing configuration.
        best_acc (float, optional): Best validation accuracy/loss recorded.
        optimizer (torch.optim.Optimizer, optional): Optimizer state.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler state.
    """
    os.makedirs(args.save_folder, exist_ok=True)
    state_dict = model.state_dict()
    save_dict = {
        'epoch': epoch,
        'best_acc': best_acc,
        'state_dict': state_dict,
    }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    filename = os.path.join(args.save_folder, f'epoch_{epoch}.pth.tar')
    torch.save(save_dict, filename)
    print('Saving checkpoint:', filename)

    best_path = os.path.join(args.save_folder, 'epoch_best.pth.tar')
    torch.save(save_dict, best_path)
    print('Saving best checkpoint:', best_path)


def train(train_loader, val_loader, model, optimizer, scheduler,
          loss_func, total_epochs, save_interval, save_folder, args):
    """
    Main training loop for the segmentation model.

    Args:
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler: Scheduler for learning rate adjustment.
        loss_func: Loss function for training/validation.
        total_epochs (int): Number of total training epochs.
        save_interval (int): Interval for saving model checkpoints.
        save_folder (str): Path to save model checkpoints.
        args (Namespace): Parsed command-line arguments.
    """
    os.makedirs(save_folder, exist_ok=True)
    val_loss_min = float('inf')

    for epoch in range(total_epochs):
        print(time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()

        scaler = GradScaler()

        train_loss = model_map(args.model).train_epoch(
            model, train_loader, optimizer,
            scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )

        print('Final training  {}/{}'.format(epoch, total_epochs - 1),
              'loss: {:.4f}'.format(train_loss),
              'time {:.2f}s'.format(time.time() - epoch_time))

        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            val_avg_loss = model_map(args.model).val_epoch(
                model, val_loader, epoch=epoch, loss_func=loss_func, args=args
            )

            print('Final validation  {}/{}'.format(epoch, total_epochs - 1),
                  'loss:', val_avg_loss,
                  'time {:.2f}s'.format(time.time() - epoch_time))

            if val_avg_loss < val_loss_min:
                print('New best ({:.6f} --> {:.6f}).'.format(val_loss_min, val_avg_loss))
                val_loss_min = val_avg_loss
                save_checkpoint(model, epoch, args,
                                best_acc=val_loss_min,
                                optimizer=optimizer, scheduler=scheduler)

        if scheduler is not None:
            scheduler.step()

    print('Training Finished! Best Validation Loss:', val_loss_min)


if __name__ == '__main__':
    args = parse_opts()
    args.phase = 'train'

    torch.manual_seed(args.manual_seed)

    model, optimizer, scheduler, loss_func = model_map(args.model).generate_model(args)

    training_dataset = dataset_map(args.dataset)(args.data_root, args.img_list, args)
    train_loader = DataLoader(training_dataset,
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)

    val_dataset = dataset_map(args.dataset)(args.data_root, args.val_list, args)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)

    train(train_loader, val_loader, model, optimizer, scheduler, loss_func,
          total_epochs=args.n_epochs, save_interval=args.save_intervals,
          save_folder=args.save_folder, args=args)

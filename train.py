from setting import model_map
from setting import dataset_map
from setting import parse_opts
from setting import evaluate

import torch
import numpy as np
import os
import time
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from scipy import ndimage
from torch.cuda.amp import autocast, GradScaler


def save_checkpoint(model,
                    epoch,
                    args,
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
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
    state_dict = model.state_dict()
    save_dict = {
        'epoch': epoch,
        'best_acc': best_acc,
        'state_dict': state_dict
    }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    filename = '{}_epoch_{}.pth.tar'.format(args.save_folder, epoch)
    torch.save(save_dict, filename)
    print('Saving checkpoint:', filename)


def train(train_loader, val_loader, model, optimizer, scheduler, loss_func, total_epochs, save_interval, save_folder, args):
    """
    Main training loop for the segmentation model.

    Args:
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler for learning rate adjustment.
        loss_func (function): Loss function for training/validation.
        total_epochs (int): Number of total training epochs.
        save_interval (int): Interval for saving model checkpoints.
        save_folder (str): Path to save model checkpoints.
        args (Namespace): Parsed command-line arguments.
    """
    val_loss_min = float('inf')  # Track best validation loss

    for epoch in range(total_epochs):
        print(time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()

        scaler = GradScaler()  # Initialize mixed precision scaler

        # One full training epoch
        train_loss = model_map(args.model).train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            loss_func=loss_func,
            args=args
        )

        print('Final training  {}/{}'.format(epoch, total_epochs - 1),
              'loss: {:.4f}'.format(train_loss),
              'time {:.2f}s'.format(time.time() - epoch_time))

        b_new_best = False  # Flag for best validation loss

        # Validation at specified intervals
        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            val_avg_loss = model_map(args.model).val_epoch(
                model,
                val_loader,
                epoch=epoch,
                loss_func=loss_func,
                args=args
            )

            print('Final validation  {}/{}'.format(epoch, total_epochs - 1),
                  'loss:', val_avg_loss,
                  'time {:.2f}s'.format(time.time() - epoch_time))

            # Save best model
            if val_avg_loss < val_loss_min:
                print('New best ({:.6f} --> {:.6f}). '.format(val_loss_min, val_avg_loss))
                val_loss_min = val_avg_loss
                b_new_best = True

                save_checkpoint(model, epoch, args,
                                best_acc=val_loss_min,
                                optimizer=optimizer,
                                scheduler=scheduler)

        # Adjust learning rate if scheduler is used
        if scheduler is not None:
            scheduler.step()

    print('Training Finished!, Best Validation Loss:', val_loss_min)


if __name__ == '__main__':
    # Parse command-line options
    args = parse_opts()
    args.phase = 'train'

    # Set random seed for reproducibility
    torch.manual_seed(args.manual_seed)

    # Load model, optimizer, scheduler, and loss function
    model, optimizer, scheduler, loss_func = model_map(args.model).generate_model(args)

    # Create training dataset and DataLoader
    training_dataset = dataset_map(args.dataset)(args.data_root, args.img_list, args)
    train_loader = DataLoader(training_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory)

    # Create validation dataset and DataLoader
    val_dataset = dataset_map(args.dataset)(args.data_root, args.val_list, args)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory)

    # Start training
    train(train_loader, val_loader, model, optimizer, scheduler, loss_func,
          total_epochs=args.n_epochs,
          save_interval=args.save_intervals,
          save_folder=args.save_folder,
          args=args)

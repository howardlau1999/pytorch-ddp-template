#!/usr/bin/env python
from typing import Dict
from torch import optim
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils import data
from model import FooModel
import sys
import os
import torch
import torch.cuda
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
import torch.distributed
import random
import numpy as np
from torch.multiprocessing import Process
from tqdm import tqdm
from tqdm.auto import trange
from utils import getLoggerWithRank, redirect_warnings_to_logger
from dataset import FooDataset
import argparse
import warnings
from utils import is_main_process

PT_LR_SCHEDULER_WARNING = "Please also save or load the state of the optimzer when saving or loading the scheduler."

def reissue_pt_warnings(caught_warnings):
    # Reissue warnings that are not the PT_LR_SCHEDULER_WARNING
    if len(caught_warnings) > 1:
        for w in caught_warnings:
            if w.category != UserWarning or w.message != PT_LR_SCHEDULER_WARNING:
                warnings.warn(w.message, w.category)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

log = None


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def save_model(model, save_directory):
    if os.path.isfile(save_directory):
        log.error("Provided path should be a directory, not a file",
                  dict(save_directory=save_directory))
        return
    os.makedirs(save_directory, exist_ok=True)

    # Only save the model itself if we are using distributed training
    model_to_save = model.module if hasattr(model, "module") else model

    state_dict = model_to_save.state_dict()
    output_model_file = os.path.join(save_directory, "model.bin")
    torch.save(state_dict, output_model_file)
    log.info("Saved model weights.", dict(path=output_model_file))


def setup(args):
    global log
    if sys.platform == 'win32':
        raise NotImplementedError("Unsupported Platform")
    else:
        args.local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
        log = getLoggerWithRank(__name__, int(
            os.environ.get("RANK", "-1")), args.local_rank)
        redirect_warnings_to_logger(log)
        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            if torch.cuda.is_available() and not args.no_cuda:
                device = torch.device("cuda")
            else:
                log.critical("!!!! Using CPU for training !!!!")
                device = torch.device("cpu")
            args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
            log.info("Using DataParallel for training.",
                     dict(n_gpu=args.n_gpu))
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            log.warning("Initializing process group.")
            torch.distributed.init_process_group(backend="nccl")
            args.node_rank = torch.distributed.get_rank()
            args.world_size = torch.distributed.get_world_size()
            log.info("Initialized distributed training process group.", dict(
                backend=torch.distributed.get_backend(), world_size=args.world_size))
            args.n_gpu = 1
        args.device = device
        args.train_batch_size = args.per_gpu_train_batch_size * \
            max(1, args.n_gpu)
        set_seed(args)

    log.warning("Finish setup.", dict(device=args.device, n_gpu=args.n_gpu,
                                      distributed_training=bool(args.local_rank != -1)))


def cleanup(args):
    if args.local_rank != -1:
        log.warning("Destroying process group.")
        torch.distributed.destroy_process_group()

def evaluate(args, model):
    pass

def train(args, model):
    # Setup TensorBoard for visualization
    if is_main_process():
        tb_writer = SummaryWriter()

    # Transfer the model to the proper device
    model = model.to(args.device)

    # Load the training dataset
    train_dataset = FooDataset(100000)

    # Choose a proper sampler
    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset
        )

    # Prepare dataloader
    train_dataloader = torch.utils.data.dataloader.DataLoader(train_dataset,
                                                              batch_size=args.train_batch_size,
                                                              sampler=train_sampler,
                                                              pin_memory=True,
                                                              )

    # Calculate total steps to train
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Loss Function, Optimizer, Scheduler
    criterion = nn.MSELoss().to(args.device)
    if args.fp16:
        try:
            from apex.optimizers import FusedAdam
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedSGD(model.parameters(),
                              lr=1e-3)
        model, optimizer = amp.initialize(
            model,
            optimizers=optimizer,
            opt_level=args.fp16_opt_level,
            keep_batchnorm_fp32=False,
            loss_scale="dynamic" if args.loss_scale == 0 else args.loss_scale,
        )
        log.info("FP16 launched")
    else:
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Wrap the model using the proper parellel strategy
    if args.n_gpu > 1:
        # Multi-GPU Training
        model = DataParallel(model)
    elif args.local_rank != -1:
        # Distributed Training
        model = DistributedDataParallel(model, device_ids=[
                                        args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    log.info("Finish setting up args.", dict(args=args.__dict__))
    log.info("Begin training.", dict(num_examples=len(train_dataset), total_batch_size=args.train_batch_size
                                                 * args.gradient_accumulation_steps
                                                 * (torch.distributed.get_world_size()
                                                    if args.local_rank != -1 else 1),
                                                 total_optimization_steps=t_total,
                                                 gradient_accumulation_steps=args.gradient_accumulation_steps))
    tr_loss, logging_loss = 0.0, 0.0
    global_step = 1

    # It is safer to use model.zero_grad() in case of multiple optimizers
    model.zero_grad()

    # Training Loop
    for epoch in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0], leave=False):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)
        with tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=args.local_rank not in [-1, 0], leave=False) as batch_iterator:
            for step, (x, y) in enumerate(batch_iterator):
                # Forward pass
                # Don't forget to call model.train()!
                model.train()
                x, y = x.to(args.device), y.to(args.device)
                outputs = model(x)
                loss = criterion(outputs, y)

                # Handling loss scaling
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # Don't call zero_grad() until we have accumulated enough steps
                loss.backward()
                batch_iterator.set_postfix(loss=loss.item())

                tr_loss += loss.item()

                # When we reached accumulation steps, do the optimization step
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                # Log metrics
                if is_main_process() and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar(
                        "lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint. It is important only one process should save the model.
                if is_main_process() and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    save_model(model, output_dir)

                    saved_args_file = os.path.join(
                        output_dir, "training_args.bin")
                    torch.save(args, saved_args_file)
                    log.info("Saved training args.",
                             dict(path=saved_args_file))

                    saved_optimizer_file = os.path.join(
                        output_dir, "optimizer.pt")
                    saved_scheduler_file = os.path.join(
                        output_dir, "scheduler.pt")
                    torch.save(optimizer.state_dict(), saved_optimizer_file)
                    log.info("Saved optimizer states.",
                             dict(path=saved_optimizer_file))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(scheduler.state_dict(), saved_scheduler_file)
                        reissue_pt_warnings(caught_warnings)
                    log.info("Saved scheduler states.",
                             dict(path=saved_scheduler_file))

                # We have done training, break loop
                if args.max_steps > 0 and global_step > args.max_steps:
                    break

        # We have done training, break loop
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    log.info("Finished training.", dict(
        global_step=global_step, average_loss=tr_loss / global_step))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1000.)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--loss_scale", type=int, default=0)
    parser.add_argument("--fp16_opt_level", type=str, default="O2")
    args = parser.parse_args()
    setup(args)
    model = FooModel()
    train(args, model)
    cleanup(args)
    log.warning("Process exited.")


if __name__ == '__main__':
    main()

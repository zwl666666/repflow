import datetime
import math
import sys
import time

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from train_utils import maybe_update_student_ema


def _format_max_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def train_one_epoch(
    model,
    model_without_ddp,
    data_loader,
    optimizer,
    device,
    epoch,
    log_writer=None,
    args=None,
    logger=None,
    train_steps_offset=0,
    student_ema=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    num_steps = len(data_loader)
    log_interval = max(1, getattr(args, "log_freq", 100) if args is not None else 100)

    optimizer.zero_grad()

    epoch_start = time.time()
    log_window_start = epoch_start
    running_flow_loss = 0.0
    running_cons_loss = 0.0
    log_steps = 0

    if misc.is_main_process() and logger is not None:
        logger.info(
            "Beginning epoch %d (%d steps, global_step=%d)",
            epoch,
            num_steps,
            train_steps_offset,
        )

    for data_iter_step, (x, labels) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        lr_sched.adjust_learning_rate(
            optimizer, data_iter_step / num_steps + epoch, args
        )

        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, loss_cons = model(x, labels, model_without_ddp)

        loss_value = (loss - loss_cons).item()
        loss_cons_value = loss_cons.item()
        if not math.isfinite(loss_value):
            msg = "Loss is {}, stopping training".format(loss_value)
            if logger is not None:
                logger.error(msg)
            else:
                print(msg)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if model_without_ddp.maybe_update_ema():
            maybe_update_student_ema(student_ema, model_without_ddp)

        metric_logger.update(loss=loss_value)
        metric_logger.update(cons_loss=loss_cons_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        cons_loss_value_reduce = misc.all_reduce_mean(loss_cons_value)

        running_flow_loss += loss_value_reduce
        running_cons_loss += cons_loss_value_reduce
        log_steps += 1
        global_step = train_steps_offset + data_iter_step + 1
        step_in_epoch = data_iter_step + 1

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / num_steps + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("train_cons_loss", cons_loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", lr, epoch_1000x)

        if misc.is_main_process() and logger is not None and (
            step_in_epoch % log_interval == 0 or step_in_epoch == num_steps
        ):
            elapsed = max(time.time() - log_window_start, 1e-6)
            steps_per_sec = log_steps / elapsed
            avg_flow = running_flow_loss / log_steps
            avg_cons = running_cons_loss / log_steps
            logger.info(
                "(epoch=%d step=%d/%d global_step=%d) "
                "flow_loss=%.4f cons_loss=%.4f total_loss=%.4f lr=%.2e steps/sec=%.2f max_mem=%.0fMB",
                epoch,
                step_in_epoch,
                num_steps,
                global_step,
                avg_flow,
                avg_cons,
                avg_flow + avg_cons,
                lr,
                steps_per_sec,
                _format_max_mem_mb(),
            )
            running_flow_loss = 0.0
            running_cons_loss = 0.0
            log_steps = 0
            log_window_start = time.time()

    metric_logger.synchronize_between_processes()
    epoch_elapsed = time.time() - epoch_start
    epoch_elapsed_str = str(datetime.timedelta(seconds=int(epoch_elapsed)))
    steps_per_sec = num_steps / max(epoch_elapsed, 1e-6)
    next_train_steps = train_steps_offset + num_steps

    if misc.is_main_process():
        summary = str(metric_logger)
        if logger is not None:
            logger.info(
                "Epoch %d finished in %s (%.2f steps/sec) | %s",
                epoch,
                epoch_elapsed_str,
                steps_per_sec,
                summary,
            )
        else:
            print(
                "Epoch {} finished in {} ({:.2f} steps/sec) | {}".format(
                    epoch, epoch_elapsed_str, steps_per_sec, summary
                )
            )

    return metric_logger, next_train_steps

"""Training and evaluation entry point for the AMSG framework."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from dataloader.dataloader import build_dataloaders
from model.model import AMSG
from utils.helper import loss as MultimodalLoss
from utils.helper import save_results
from utils.metrics import f1, macro, micro


LOGGER = logging.getLogger("amsg")


class WarmupCosineScheduler(LambdaLR):
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 1e-5,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = max(0, warmup_epochs)
        self.total_epochs = max(1, total_epochs)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, current_epoch: int) -> float:
        if self.warmup_epochs > 0 and current_epoch < self.warmup_epochs:
            return current_epoch / float(self.warmup_epochs)

        decay_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = (current_epoch - self.warmup_epochs) / float(decay_epochs)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + cosine * (1.0 - self.min_lr_ratio)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_logger(log_path: Path) -> logging.Logger:
    """Configure console and file logging for one training run."""

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    return LOGGER


def build_metric_buffers(args, ner2idx, rel2idx):
    """Create count buffers used by the selected evaluation metric."""

    entity_counts = [0, 0, 0]
    relation_counts = [0, 0, 0]
    triple_counts = [0, 0, 0]

    if args.eval_metric == "macro":
        entity_counts *= len(ner2idx)
        relation_counts *= len(rel2idx)
        triple_counts *= len(rel2idx)

    return entity_counts, relation_counts, triple_counts


def add_counts(total, current) -> None:
    """Add one metric-count vector to an accumulated vector."""

    for index, value in enumerate(current):
        total[index] += value


def move_to_device(value, device):
    """Move tensors to the selected device while allowing ``None`` labels."""

    return value.to(device) if hasattr(value, "to") else value


def collect_task_counts(
    args,
    metric,
    entity_counts,
    relation_counts,
    triple_counts,
    entity_predictions,
    relation_predictions,
    entity_labels,
    relation_labels,
) -> None:
    """Update the metric buffers for the selected dataset."""

    if args.data == "MNRE":
        add_counts(
            relation_counts,
            metric.count_re_num(relation_predictions, relation_labels),
        )
    elif args.data in {"twitter15", "twitter17"}:
        add_counts(
            entity_counts,
            metric.count_ner_num(entity_predictions, entity_labels),
        )
    else:
        add_counts(
            entity_counts,
            metric.count_ner_num(entity_predictions, entity_labels),
        )
        add_counts(
            triple_counts,
            metric.count_num(
                entity_predictions,
                entity_labels,
                relation_predictions,
                relation_labels,
            ),
        )


def finalize_metrics(args, entity_counts, relation_counts, triple_counts):
    """Convert accumulated counts to precision, recall, and F1 dictionaries."""

    results = {"entity": None, "relation": None, "triple": None}
    if args.data == "MNRE":
        results["relation"] = f1(relation_counts)
    elif args.data in {"twitter15", "twitter17"}:
        results["entity"] = f1(entity_counts)
    else:
        results["entity"] = f1(entity_counts)
        results["triple"] = f1(triple_counts)
    return results


def log_split_results(split_name: str, results: Dict[str, Optional[Dict[str, float]]], loss_value: float) -> None:
    """Write evaluation results to the run logger."""

    LOGGER.info("------ %s Results ------", split_name)
    LOGGER.info("loss: %.4f", loss_value)
    for name, result in results.items():
        if result is not None:
            LOGGER.info(
                "%s: p=%.4f, r=%.4f, f=%.4f",
                name,
                result["p"],
                result["r"],
                result["f"],
            )


def evaluate_split(
    data_loader,
    model,
    criterion,
    metric,
    ner2idx,
    rel2idx,
    args,
    device,
    split_name: str,
):
    """Evaluate AMSG on one data split."""

    entity_counts, relation_counts, triple_counts = build_metric_buffers(
        args,
        ner2idx,
        rel2idx,
    )
    total_loss = 0.0
    steps = 0

    with torch.inference_mode():
        for batch in data_loader:
            steps += 1
            texts = batch[0]
            entity_labels = move_to_device(batch[2], device)
            relation_labels = move_to_device(batch[3], device)
            mask = batch[4].to(device)
            images = batch[5].to(device)
            sentences = batch[6]

            entity_predictions, relation_predictions = model(
                texts,
                images,
                sentences,
                mask,
            )
            batch_loss = criterion(
                entity_predictions,
                entity_labels,
                relation_predictions,
                relation_labels,
            )
            total_loss += batch_loss.item()

            collect_task_counts(
                args,
                metric,
                entity_counts,
                relation_counts,
                triple_counts,
                entity_predictions,
                relation_predictions,
                entity_labels,
                relation_labels,
            )

    average_loss = total_loss / max(steps, 1)
    results = finalize_metrics(args, entity_counts, relation_counts, triple_counts)
    log_split_results(split_name, results, average_loss)
    results["loss"] = average_loss
    return results


def train_one_epoch(
    data_loader,
    model,
    optimizer,
    criterion,
    metric,
    ner2idx,
    rel2idx,
    args,
    device,
    epoch: int,
):
    """Train AMSG for one epoch."""

    model.train()
    entity_counts, relation_counts, triple_counts = build_metric_buffers(
        args,
        ner2idx,
        rel2idx,
    )
    running_loss = 0.0
    steps = 0

    for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
        steps += 1
        optimizer.zero_grad(set_to_none=True)

        texts = batch[0]
        entity_labels = move_to_device(batch[2], device)
        relation_labels = move_to_device(batch[3], device)
        mask = batch[4].to(device)
        images = batch[5].to(device)
        sentences = batch[6]

        entity_predictions, relation_predictions = model(
            texts,
            images,
            sentences,
            mask,
        )
        batch_loss = criterion(
            entity_predictions,
            entity_labels,
            relation_predictions,
            relation_labels,
        )
        if model.last_auxiliary_loss is not None:
            batch_loss = batch_loss + model.last_auxiliary_loss

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        running_loss += batch_loss.item()
        collect_task_counts(
            args,
            metric,
            entity_counts,
            relation_counts,
            triple_counts,
            entity_predictions,
            relation_predictions,
            entity_labels,
            relation_labels,
        )

        if steps % args.log_steps == 0:
            LOGGER.info(
                "Epoch: %d, step: %d/%d, loss=%.4f, lr=%.8f",
                epoch,
                steps,
                len(data_loader),
                running_loss / steps,
                optimizer.param_groups[0]["lr"],
            )

    average_loss = running_loss / max(steps, 1)
    results = finalize_metrics(args, entity_counts, relation_counts, triple_counts)
    log_split_results("Training", results, average_loss)
    results["loss"] = average_loss
    return results


def selection_score(results) -> float:
    """Return the development score used for checkpoint selection."""

    return sum(
        result["f"]
        for name, result in results.items()
        if name in {"entity", "relation", "triple"} and result is not None
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Train and evaluate AMSG.")
    parser.add_argument("--data", required=True, choices=["JMERE", "MNRE", "twitter15", "twitter17"])
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--hidden_size", type=int, default=1100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--bert_local_path", required=True)
    parser.add_argument("--blip_local_path", required=True)
    parser.add_argument(
        "--clip_local_path",
        default="openai/clip-vit-base-patch32",
        help="Local CLIP directory or a Hugging Face model identifier.",
    )
    parser.add_argument("--eval_metric", choices=["micro", "macro"], default="micro")
    parser.add_argument("--sim_mode", choices=["itc", "itm"], default="itc")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--linear_warmup_rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropconnect", type=float, default=0.1)
    parser.add_argument("--steps", dest="log_steps", type=int, default=100)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--clip", dest="max_grad_norm", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    output_dir = Path("save") / args.output_file
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_dir / f"{args.output_file}.log")
    logger.info("Command: %s", sys.argv)
    logger.info("Arguments: %s", args)
    logger.info("Device: %s", device)

    with open(Path("datasets") / args.data / "ner2idx.json", encoding="utf-8") as file:
        ner2idx = json.load(file)
    with open(Path("datasets") / args.data / "rel2idx.json", encoding="utf-8") as file:
        rel2idx = json.load(file)

    train_loader, test_loader, dev_loader = build_dataloaders(args, ner2idx, rel2idx)
    model = AMSG(args, input_size=768, ner2idx=ner2idx, rel2idx=rel2idx).to(device)

    criterion = MultimodalLoss()
    metric = micro(rel2idx, ner2idx) if args.eval_metric == "micro" else macro(rel2idx, ner2idx)

    if not args.do_train and not args.do_eval:
        logger.info("Neither --do_train nor --do_eval was specified. Nothing to run.")
        return

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_epochs = (
        min(args.epoch, max(1, int(args.epoch * args.linear_warmup_rate)))
        if args.linear_warmup_rate > 0
        else min(args.epoch, 5)
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=args.epoch,
    )

    result_file = save_results(
        str(output_dir / f"{args.output_file}.txt"),
        header="# epoch\ttrain_loss\tdev_loss\ttest_loss",
    )
    best_score = float("-inf")
    best_test_results = None

    for epoch in range(args.epoch):
        if args.do_train:
            train_one_epoch(
                train_loader,
                model,
                optimizer,
                criterion,
                metric,
                ner2idx,
                rel2idx,
                args,
                device,
                epoch,
            )

        if args.do_eval:
            model.eval()
            dev_results = evaluate_split(
                dev_loader,
                model,
                criterion,
                metric,
                ner2idx,
                rel2idx,
                args,
                device,
                "Development",
            )
            test_results = evaluate_split(
                test_loader,
                model,
                criterion,
                metric,
                ner2idx,
                rel2idx,
                args,
                device,
                "Test",
            )

            current_score = selection_score(dev_results)
            if current_score > best_score:
                best_score = current_score
                best_test_results = test_results
                torch.save(model.state_dict(), output_dir / f"{args.output_file}.pt")
                logger.info("Best development checkpoint saved.")

            result_file.save(
                f"{epoch}\t{dev_results['loss']:.4f}\t{test_results['loss']:.4f}"
            )

        scheduler.step()

    if args.do_train:
        torch.save(model.state_dict(), output_dir / "final.pt")
        logger.info("Final checkpoint saved.")

    if best_test_results is not None:
        result_file.save("Best test results:")
        for name, result in best_test_results.items():
            if name in {"entity", "relation", "triple"} and result is not None:
                result_file.save(
                    f"{name}: p={result['p']:.4f}\tr={result['r']:.4f}\tf={result['f']:.4f}"
                )


if __name__ == "__main__":
    main()

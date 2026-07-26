"""Embed v1 training with cached in-batch negatives."""

from __future__ import annotations

import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from dataset import attach_mined_negatives, load_positive_groups, make_no_duplicate_batches, sample_one_positive_per_query
from data.load import load_retrieval_split
from evaluate.metrics import evaluate_rankings
from callback import WandbTrainingCallback


class EmbedTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config["random_seed"])
        self.output_dir = Path(config["output_directory"])
        self.report_path = Path(config["report_path"])

    def train(self):
        try:
            import numpy as np
            import torch
            from sentence_transformers import InputExample, SentenceTransformer, losses, models, util
            from transformers import get_linear_schedule_with_warmup
        except ImportError as exc:
            raise RuntimeError("Install the project training dependencies with: pip install -e '.[train]'") from exc

        self._validate_data_paths()
        self._seed_everything(torch, np)
        device = self._resolve_device(torch)
        model = SentenceTransformer(self.config["model_name"], device=device, trust_remote_code=True)
        if not self.config.get("mixed_precision", False):
            # GradCache recomputes mini-batches during backward. Representations
            # and cached gradients must have the same dtype; disabling AMP does
            # not itself upcast a model loaded from half-precision weights.
            model.float()
        model.max_seq_length = int(self.config["sequence_length"])
        if not any(isinstance(module, models.Normalize) for module in model._modules.values()):
            model.add_module("normalize", models.Normalize())
        first_parameter = next(model.parameters())
        print(
            f"[setup] model={self.config['model_name']} device={model.device} "
            f"dtype={first_parameter.dtype} "
            f"mixed_precision={self.config.get('mixed_precision', False)}",
            flush=True,
        )
        queries, corpus, positives = load_positive_groups(self._path("train_queries").parent)
        epochs = int(self.config["epochs"])
        batch_size = int(self.config["batch_size"])
        preview_pairs = sample_one_positive_per_query(queries, corpus, positives, self.seed, 0)
        if self.config.get("hard_negatives_path"):
            preview_pairs = attach_mined_negatives(
                preview_pairs, corpus, positives, self._path("hard_negatives_path"),
                int(self.config["hard_negatives_per_query"]),
            )
        steps_per_epoch = (len(make_no_duplicate_batches(preview_pairs, batch_size))
                           if self.config.get("sampling", {}).get("no_duplicates_per_batch", True)
                           else math.ceil(len(preview_pairs) / batch_size))
        total_steps = steps_per_epoch * epochs
        warmup_steps = round(total_steps * float(self.config["warmup_ratio"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config["learning_rate"]),
                                      weight_decay=float(self.config["weight_decay"]))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        loss_model = losses.CachedMultipleNegativesRankingLoss(
            model=model, mini_batch_size=int(self.config["mini_batch_size"]),
            similarity_fct=util.dot_score, scale=float(self.config["loss_scale"])
        )
        if self.config.get("mixed_precision", False):
            raise ValueError(
                "Embed v1 uses CachedMultipleNegativesRankingLoss and requires "
                "mixed_precision: false to keep GradCache representations and gradients in one dtype"
            )
        history, started = [], time.perf_counter()
        tracker = WandbTrainingCallback(self.config.get("tracking", {}), self.config)
        global_step = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            model.train()
            pairs = sample_one_positive_per_query(queries, corpus, positives, self.seed, epoch)
            if self.config.get("hard_negatives_path"):
                pairs = attach_mined_negatives(
                    pairs, corpus, positives, self._path("hard_negatives_path"),
                    int(self.config["hard_negatives_per_query"]),
                )
            if self.config.get("sampling", {}).get("no_duplicates_per_batch", True):
                pair_batches = make_no_duplicate_batches(pairs, batch_size)
            else:
                pair_batches = [pairs[index:index + batch_size]
                                for index in range(0, len(pairs), batch_size)]
            losses_this_epoch = []
            for batch_index, pair_batch in enumerate(pair_batches, 1):
                examples = [InputExample(texts=[pair.query, pair.passage, *pair.negative_passages])
                            for pair in pair_batch]
                features, labels = model.smart_batching_collate(examples)
                features = [util.batch_to_device(feature, model.device) for feature in features]
                if labels is not None and hasattr(labels, "to"):
                    labels = labels.to(model.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_model(features, labels)
                if not torch.isfinite(loss).all():
                    query_ids = [pair.query_id for pair in pair_batch]
                    raise FloatingPointError(
                        "Non-finite training loss before backward: "
                        f"epoch={epoch + 1}, batch={batch_index}, "
                        f"step={global_step + 1}, loss={loss.detach().cpu().item()}, "
                        f"model_dtype={first_parameter.dtype}, device={model.device}, "
                        f"query_ids={query_ids}"
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(self.config["max_grad_norm"]),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                scheduler.step()
                loss_value = float(loss.detach().cpu())
                losses_this_epoch.append(loss_value)
                global_step += 1
                learning_rate = optimizer.param_groups[0]["lr"]
                print(
                    f"[train] run={self.config.get('run_name', 'embed')} "
                    f"epoch={epoch + 1}/{epochs} "
                    f"batch={batch_index}/{len(pair_batches)} "
                    f"step={global_step} loss={loss_value:.6f} "
                    f"grad_norm={float(gradient_norm.detach().cpu()):.6f} "
                    f"lr={learning_rate:.8g}",
                    flush=True,
                )
                tracker.log_train_step(global_step, epoch + 1, batch_index, loss_value,
                                       learning_rate, torch)

            validation = self._evaluate(model, Path(self.config["validation_queries"]).parent,
                                        int(self.config["validation_batch_size"]))
            epoch_record = {"epoch": epoch + 1,
                            "mean_training_loss": sum(losses_this_epoch) / len(losses_this_epoch),
                            "validation": validation}
            history.append(epoch_record)
            tracker.log_validation(global_step, epoch + 1, validation,
                                   epoch_record["mean_training_loss"])
            if (epoch + 1) % int(self.config["checkpoint_every_epochs"]) == 0:
                model.save_pretrained(str(self.output_dir / f"checkpoint-epoch-{epoch + 1}"))
            self._write_report(history, device, time.perf_counter() - started, complete=False)

        model.save_pretrained(str(self.output_dir / "final"))
        report = self._write_report(history, device, time.perf_counter() - started, complete=True)
        tracker.finish()
        return report

    def _evaluate(self, model, split_dir: Path, batch_size: int):
        import numpy as np
        queries, corpus, qrels = load_retrieval_split(split_dir)
        query_ids, passage_ids = list(queries), list(corpus)
        encode_query = getattr(model, "encode_query", model.encode)
        encode_document = getattr(model, "encode_document", model.encode)
        qvec = encode_query([queries[key] for key in query_ids], batch_size=batch_size,
                            normalize_embeddings=True, convert_to_numpy=True)
        pvec = encode_document([corpus[key] for key in passage_ids], batch_size=batch_size,
                               normalize_embeddings=True, convert_to_numpy=True)
        order = np.argsort(-(qvec @ pvec.T), axis=1)
        rankings = {qid: [passage_ids[index] for index in order[row]]
                    for row, qid in enumerate(query_ids)}
        return evaluate_rankings(rankings, qrels)

    def _seed_everything(self, torch, np):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _resolve_device(self, torch):
        configured = str(self.config.get("device", "auto"))
        return ("cuda" if torch.cuda.is_available() else "cpu") if configured == "auto" else configured

    def _path(self, key):
        path = Path(self.config[key])
        if not path.exists():
            raise FileNotFoundError(f"Configured path does not exist: {path}")
        return path

    def _validate_data_paths(self):
        required = [
            "train_queries", "train_corpus", "train_qrels",
            "validation_queries", "validation_corpus", "validation_qrels",
            "test_queries", "test_corpus", "test_qrels",
        ]
        if self.config.get("hard_negatives_path"):
            required.append("hard_negatives_path")
        missing_keys = [key for key in required if key not in self.config]
        if missing_keys:
            raise ValueError(f"Training config is missing dataset paths: {', '.join(missing_keys)}")
        for key in required:
            self._path(key)

    def _write_report(self, history, device, elapsed, complete):
        report = {"schema_version": 1, "run": self.config.get("run_name", "embed_v1"), "complete": complete,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "model": self.config["model_name"], "device": device,
                  "training_supervision": self.config.get(
                      "training_supervision", "positive pairs + cached in-batch negatives"),
                  "config": self.config, "history": history,
                  "elapsed_seconds": round(elapsed, 3)}
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.report_path)
        return report

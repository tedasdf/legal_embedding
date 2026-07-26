
import argparse
import random, time, os, json
from pathlib import Path

from evaluate.metrics import evaluate_rankings
from utils.config import load_config
from utils.utils import _read_jsonl
from data.load import load_retrieval_split
from callback import WandbTrainingCallback
from datetime import datetime, timezone

def _validate_teacher_records(records, queries, corpus):
    for row in records:
        if row["query_id"] not in queries: raise ValueError(f"Unknown teacher query: {row['query_id']}")
        if not row["candidates"] or not row["candidates"][0].get("is_positive"):
            raise ValueError(f"Positive candidate must be first for {row['query_id']}")
        if any(item["passage_id"] not in corpus for item in row["candidates"]):
            raise ValueError(f"Unknown teacher candidate for {row['query_id']}")
    sizes = {len(row["candidates"]) for row in records}
    if len(sizes) != 1: raise ValueError("Every teacher record must have the same candidate count")



def _evaluate_model(model, split_dir, batch_size):
    import numpy as np
    queries, corpus, qrels = load_retrieval_split(split_dir)
    qids, pids = list(queries), list(corpus)
    qvec = getattr(model, "encode_query", model.encode)([queries[q] for q in qids], batch_size=batch_size,
                                                         normalize_embeddings=True, convert_to_numpy=True)
    pvec = getattr(model, "encode_document", model.encode)([corpus[p] for p in pids], batch_size=batch_size,
                                                            normalize_embeddings=True, convert_to_numpy=True)
    order = np.argsort(-(qvec @ pvec.T), axis=1)
    return evaluate_rankings({qid: [pids[index] for index in order[row]] for row, qid in enumerate(qids)}, qrels)



class DistillationTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config["random_seed"])
        self.output_dir = Path(config["output_directory"])
        self.report_path = Path(config["report_path"])

    def train(self):
        try:
            import numpy as np
            import torch
            import torch.nn.functional as functional
            from sentence_transformers import SentenceTransformer, models, util
            from transformers import get_linear_schedule_with_warmup
        except ImportError as exc:
            raise RuntimeError("Distillation requires the project training dependencies") from exc
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(self.seed)
        device = "cuda" if self.config.get("device", "auto") == "auto" and torch.cuda.is_available() else self.config.get("device", "cpu")
        model = SentenceTransformer(self.config["model_name"], device=device, trust_remote_code=True)
        if not self.config.get("mixed_precision", False):
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
        queries, corpus, _ = load_retrieval_split(Path(self.config["train_queries"]).parent)
        records = list(_read_jsonl(Path(self.config["teacher_scores_path"])))
        _validate_teacher_records(records, queries, corpus)
        batch_size = int(self.config["batch_size"])
        mini_batch_size = int(self.config.get("mini_batch_size", batch_size))
        epochs = int(self.config["epochs"])
        if batch_size < 1 or mini_batch_size < 1:
            raise ValueError("batch_size and mini_batch_size must be positive")
        if mini_batch_size > batch_size:
            raise ValueError("mini_batch_size cannot exceed batch_size")
        print(
            f"[setup] effective_batch_size={batch_size} "
            f"mini_batch_size={mini_batch_size} "
            f"gradient_accumulation_steps="
            f"{(batch_size + mini_batch_size - 1) // mini_batch_size}",
            flush=True,
        )
        steps_per_epoch = (len(records) + batch_size - 1) // batch_size
        total_steps = steps_per_epoch * epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config["learning_rate"]),
                                      weight_decay=float(self.config["weight_decay"]))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, round(total_steps * float(self.config["warmup_ratio"])), total_steps)
        teacher_temperature = float(self.config["teacher_temperature"])
        student_scale = float(self.config["student_scale"])
        alpha = float(self.config["distillation_weight"])
        history, started = [], time.perf_counter(); self.output_dir.mkdir(parents=True, exist_ok=True)
        tracker = WandbTrainingCallback(self.config.get("tracking", {}), self.config)
        global_step = 0
        for epoch in range(epochs):
            model.train(); random.Random(f"{self.seed}:{epoch}").shuffle(records); epoch_losses = []
            for batch_index, offset in enumerate(range(0, len(records), batch_size), 1):
                logical_batch = records[offset:offset + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logical_loss = 0.0
                mini_batches = (
                    len(logical_batch) + mini_batch_size - 1
                ) // mini_batch_size
                for mini_index, mini_offset in enumerate(
                    range(0, len(logical_batch), mini_batch_size),
                    1,
                ):
                    batch = logical_batch[
                        mini_offset:mini_offset + mini_batch_size
                    ]
                    candidate_count = len(batch[0]["candidates"])
                    query_texts = [queries[row["query_id"]] for row in batch]
                    document_texts = [
                        corpus[item["passage_id"]]
                        for row in batch
                        for item in row["candidates"]
                    ]
                    qfeatures = util.batch_to_device(
                        model.tokenize(query_texts),
                        model.device,
                    )
                    dfeatures = util.batch_to_device(
                        model.tokenize(document_texts),
                        model.device,
                    )
                    qemb = model(qfeatures)["sentence_embedding"]
                    demb = model(dfeatures)["sentence_embedding"].reshape(
                        len(batch),
                        candidate_count,
                        -1,
                    )
                    student_logits = (
                        torch.einsum("bd,bcd->bc", qemb, demb)
                        * student_scale
                    )
                    teacher_logits = torch.tensor(
                        [
                            [
                                item["teacher_score"]
                                for item in row["candidates"]
                            ]
                            for row in batch
                        ],
                        device=model.device,
                        dtype=student_logits.dtype,
                    )
                    teacher_distribution = functional.softmax(
                        teacher_logits / teacher_temperature,
                        dim=1,
                    )
                    distill_loss = functional.kl_div(
                        functional.log_softmax(student_logits, dim=1),
                        teacher_distribution,
                        reduction="batchmean",
                    )
                    positive_targets = torch.zeros(
                        len(batch),
                        dtype=torch.long,
                        device=model.device,
                    )
                    contrastive_loss = functional.cross_entropy(
                        student_logits,
                        positive_targets,
                    )
                    loss = (
                        alpha * distill_loss
                        + (1.0 - alpha) * contrastive_loss
                    )
                    if not torch.isfinite(loss).all():
                        query_ids = [item["query_id"] for item in batch]
                        raise FloatingPointError(
                            "Non-finite distillation loss before backward: "
                            f"epoch={epoch + 1}, batch={batch_index}, "
                            f"mini_batch={mini_index}/{mini_batches}, "
                            f"step={global_step + 1}, "
                            f"loss={loss.detach().cpu().item()}, "
                            f"distill_loss="
                            f"{distill_loss.detach().cpu().item()}, "
                            f"contrastive_loss="
                            f"{contrastive_loss.detach().cpu().item()}, "
                            f"model_dtype={first_parameter.dtype}, "
                            f"device={model.device}, query_ids={query_ids}"
                        )
                    # Both component losses are means over the mini-batch.
                    # Weight by its share of the logical batch so accumulated
                    # gradients equal the full logical-batch mean.
                    mini_weight = len(batch) / len(logical_batch)
                    (loss * mini_weight).backward()
                    logical_loss += float(loss.detach().cpu()) * mini_weight
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(self.config["max_grad_norm"]),
                    error_if_nonfinite=True,
                )
                optimizer.step(); scheduler.step(); loss_value = logical_loss
                epoch_losses.append(loss_value); global_step += 1
                learning_rate = optimizer.param_groups[0]["lr"]
                print(
                    f"[train] run={self.config.get('run_name', 'embed_v4')} "
                    f"epoch={epoch + 1}/{epochs} "
                    f"batch={batch_index}/{steps_per_epoch} "
                    f"mini_batches={mini_batches} "
                    f"step={global_step} loss={loss_value:.6f} "
                    f"grad_norm={float(gradient_norm.detach().cpu()):.6f} "
                    f"lr={learning_rate:.8g}",
                    flush=True,
                )
                tracker.log_train_step(global_step, epoch + 1, batch_index, loss_value,
                                       learning_rate, torch)
            validation = _evaluate_model(model, Path(self.config["validation_queries"]).parent,
                                         int(self.config["validation_batch_size"]))
            history.append({"epoch": epoch + 1, "mean_training_loss": sum(epoch_losses) / len(epoch_losses),
                            "validation": validation})
            tracker.log_validation(global_step, epoch + 1, validation,
                                   history[-1]["mean_training_loss"])
            model.save_pretrained(str(self.output_dir / f"checkpoint-epoch-{epoch + 1}"))
            self._report(history, device, time.perf_counter() - started, False)
        model.save_pretrained(str(self.output_dir / "final"))
        report = self._report(history, device, time.perf_counter() - started, True)
        tracker.finish()
        return report

    def _report(self, history, device, elapsed, complete):
        report = {"schema_version": 1, "run": "embed_v4_distillation", "complete": complete,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": self.config["model_name"],
                  "device": device, "training_supervision": "teacher distribution + positive contrastive target",
                  "config": self.config, "history": history, "elapsed_seconds": round(elapsed, 3)}
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.report_path)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embed/train_v4_distillation.yaml")
    args = parser.parse_args()
    report = DistillationTrainer(load_config(args.config)).train()
    print(f"Training complete. Final validation: {report['history'][-1]['validation']}")


if __name__ == "__main__": main()

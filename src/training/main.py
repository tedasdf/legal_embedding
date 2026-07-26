#!/usr/bin/env python3
"""Train an Embed model from a reproducible YAML configuration."""

from __future__ import annotations

import argparse

from utils.config import load_config
from trainer import EmbedTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embed/train.yaml")
    args = parser.parse_args()
    trainer = EmbedTrainer(load_config(args.config))
    report = trainer.train()
    print(f"Training complete. Final validation: {report['history'][-1]['validation']}")


if __name__ == "__main__":
    main()

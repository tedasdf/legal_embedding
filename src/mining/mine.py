import argparse
import json

from bm25 import mine_dense_hard_negatives
from dense import mine_bm25_hard_negatives
from utils.config import load_config



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embed/mine_bm25.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("method") == "bm25":
        report = mine_bm25_hard_negatives(config)
    elif config.get("method") == "dense":
        report = mine_dense_hard_negatives(config)
    print(json.dumps(report["statistics"], indent=2))
    

if __name__ == "__main__":
    main()

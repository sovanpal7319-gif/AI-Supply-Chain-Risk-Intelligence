"""
SPERT Fine-Tuning Script for Supply-Chain Domain

Trains a SpERT model on custom supply-chain disruption annotated data.
Initializes from bert-base-cased and saves to models/spert_supply_chain/.

Usage:
  # From project root:
  python scripts/train_spert.py

  # Or with custom config:
  python scripts/train_spert.py --config data/spert/supply_chain_train.conf

  # Or run SPERT directly:
  cd spert
  python spert.py train --config ../data/spert/supply_chain_train.conf

After training, the model will be saved to models/spert_supply_chain/
and automatically loaded by the SpertAgent on next server start.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root and spert root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SPERT_ROOT = _PROJECT_ROOT / "spert"

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SPERT_ROOT))


def train_spert(config_path: str = None):
    """
    Fine-tune SpERT on supply-chain disruption data.

    Parameters
    ----------
    config_path : str, optional
        Path to .conf file. Defaults to data/spert/supply_chain_train.conf
    """
    from spert.spert_trainer import SpERTTrainer
    from spert.input_reader import JsonInputReader

    if config_path is None:
        config_path = str(_PROJECT_ROOT / "data" / "spert" / "supply_chain_train.conf")

    # Parse config
    from spert.config_reader import process_configs
    from args import train_argparser

    arg_parser = train_argparser()

    def _run_training(run_args):
        """Callback for process_configs."""
        print("\n" + "=" * 60)
        print("  SPERT Supply-Chain Fine-Tuning")
        print("=" * 60)
        print(f"  Model:     {run_args.model_path}")
        print(f"  Train:     {run_args.train_path}")
        print(f"  Valid:     {run_args.valid_path}")
        print(f"  Types:     {run_args.types_path}")
        print(f"  Epochs:    {run_args.epochs}")
        print(f"  Batch:     {run_args.train_batch_size}")
        print(f"  LR:        {run_args.lr}")
        print(f"  Save to:   {run_args.save_path}")
        print("=" * 60 + "\n")

        trainer = SpERTTrainer(run_args)
        trainer.train(
            train_path=run_args.train_path,
            valid_path=run_args.valid_path,
            types_path=run_args.types_path,
            input_reader_cls=JsonInputReader,
        )

        print("\n" + "=" * 60)
        print("  ✅ Training complete!")
        print(f"  Model saved to: {run_args.save_path}")
        print("=" * 60)
        print("\nTo use the trained model:")
        print("  1. Copy the 'final_model' directory from the save path")
        print(f"     to: models/spert_supply_chain/")
        print("  2. Restart the server")
        print("  3. The SpertAgent will auto-detect and load it")

    # Override sys.argv to pass the config
    original_argv = sys.argv
    sys.argv = ["train_spert.py", "--config", config_path]

    try:
        process_configs(target=_run_training, arg_parser=arg_parser)
    finally:
        sys.argv = original_argv


def predict_example():
    """Run a quick prediction with a trained model to verify it works."""
    from backend.agents.spert_agent import SpertAgent

    agent = SpertAgent()

    test_sentences = [
        "Flooding in China disrupted Tesla battery suppliers",
        "Earthquake in Taiwan forced TSMC to halt chip production",
        "Samsung semiconductor factory fire in South Korea delays Apple iPhone production",
        "US sanctions against China threaten NVIDIA GPU exports",
        "Suez Canal blockade disrupts Maersk and global shipping routes",
    ]

    print("\n" + "=" * 60)
    print("  SPERT Extraction Examples")
    print("=" * 60)

    for sentence in test_sentences:
        print(f"\n📰 Input: \"{sentence}\"")
        result = agent.extract(sentence)

        print("  Entities:")
        for ent in result["entities"]:
            print(f"    [{ent['type']}] {ent['text']} (conf={ent.get('confidence', 'N/A')})")

        print("  Relations:")
        for rel in result["relations"]:
            print(f"    {rel['head']} --[{rel['type']}]--> {rel['tail']} (conf={rel.get('confidence', 'N/A')})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPERT Supply-Chain Fine-Tuning")
    parser.add_argument(
        "mode",
        nargs="?",
        default="predict",
        choices=["train", "predict"],
        help="Mode: 'train' to fine-tune, 'predict' to run examples (default: predict)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to SPERT training .conf file",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_spert(args.config)
    else:
        predict_example()

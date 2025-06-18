#!/usr/bin/env python3
"""
Command-line interface for the Molecular Search Pipeline.

This module provides a user-friendly CLI for running the pipeline
with various options and configurations.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Optional

from molsearch_pipeline import MolecularSearchPipeline, Config


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file or use defaults.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object
    """
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert dictionary to Config object
        config = Config()

        # Update config with YAML values
        if "data" in config_dict:
            config.data_dir = Path(config_dict["data"]["data_dir"])
            config.models_dir = Path(config_dict["data"]["models_dir"])
            config.output_dir = Path(config_dict["data"]["output_dir"])

        if "model" in config_dict:
            config.coati_model_url = config_dict["model"]["coati_model_url"]
            config.featurizer_type = config_dict["model"]["featurizer_type"]

        if "generation" in config_dict:
            config.num_variations = config_dict["generation"]["num_variations"]
            config.noise_scale = config_dict["generation"]["noise_scale"]
            config.max_zinc_samples = config_dict["generation"]["max_zinc_samples"]

        if "pinecone" in config_dict:
            config.pinecone_api_key = config_dict["pinecone"]["api_key"]
            config.pinecone_index_name = config_dict["pinecone"]["index_name"]

        return config
    else:
        return Config()


def validate_smiles(smiles: str) -> bool:
    """
    Validate SMILES string.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Molecular Search Pipeline - Find similar molecules using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python cli.py --smiles "CCO"
  
  # Run with custom configuration
  python cli.py --smiles "CCO" --config custom_config.yaml
  
  # Run with specific parameters
  python cli.py --smiles "CCO" --variations 50 --noise 0.25
  
  # Run in verbose mode
  python cli.py --smiles "CCO" --verbose
        """,
    )

    # Required arguments
    parser.add_argument(
        "--smiles", "-s", type=str, required=True, help="Reference SMILES string"
    )

    # Optional arguments
    parser.add_argument(
        "--config", "-c", type=str, help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--variations",
        "-v",
        type=int,
        help="Number of molecular variations to generate",
    )

    parser.add_argument(
        "--noise", "-n", type=float, help="Noise scale for generation (0.0-1.0)"
    )

    parser.add_argument(
        "--featurizer",
        "-f",
        choices=["rdkit_2d", "chemgpt", "graphormer"],
        help="Molecular featurization method",
    )

    parser.add_argument(
        "--output-dir", "-o", type=str, help="Output directory for results"
    )

    parser.add_argument(
        "--verbose", "-V", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    parser.add_argument(
        "--version", action="version", version="Molecular Search Pipeline v1.0.0"
    )

    args = parser.parse_args()

    # Validate SMILES
    if not validate_smiles(args.smiles):
        print(f"Error: Invalid SMILES string: {args.smiles}")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.variations:
        config.num_variations = args.variations
    if args.noise:
        config.noise_scale = args.noise
    if args.featurizer:
        config.featurizer_type = args.featurizer
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    # Set up logging
    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    # Dry run mode
    if args.dry_run:
        print("=== DRY RUN MODE ===")
        print(f"Reference SMILES: {args.smiles}")
        print(f"Configuration: {config}")
        print("Would run the complete pipeline...")
        return

    # Run pipeline
    try:
        print(f"Starting molecular search for: {args.smiles}")
        print(
            f"Configuration: {config.featurizer_type} featurizer, "
            f"{config.num_variations} variations, noise={config.noise_scale}"
        )

        pipeline = MolecularSearchPipeline(config)
        analysis, results = pipeline.run_full_pipeline(args.smiles)

        # Print results
        print("\n=== RESULTS ===")
        print(f"Generated molecules: {analysis['num_results']}")
        print(f"Mean similarity: {analysis['mean_similarity']:.3f}")
        print(f"Max similarity: {analysis['max_similarity']:.3f}")
        print(f"High similarity (>0.7): {analysis['high_similarity_count']}")
        print(f"Medium similarity (0.5-0.7): {analysis['medium_similarity_count']}")
        print(f"Low similarity (<0.5): {analysis['low_similarity_count']}")

        print(f"\nResults saved to: {config.output_dir}")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

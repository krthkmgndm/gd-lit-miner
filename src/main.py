import argparse
import logging
from pathlib import Path
import sys
from typing import Optional
import pandas as pd
from datetime import datetime

from .utils.constants import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    GENERATED_DATA_DIR,
)
from .utils.silicon_optimizer import M2Optimizer
from .data_preprocessing import DataPreprocessor
from .nlp_model import BiomedicalNLP
from .extraction import AssociationExtractor
from .visualization import ResultsVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"mining_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class LiteratureMining:
    def __init__(self, input_dir: Optional[Path] = None):
        """Initialize the literature mining pipeline."""
        logger.info("Initializing Literature Mining Pipeline")

        # Initialize directories
        self.input_dir = input_dir or RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.reports_dir = REPORTS_DIR
        self.generated_dir = GENERATED_DATA_DIR

        # Create directories if they don't exist
        for directory in [self.processed_dir, self.reports_dir, self.generated_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize M2 optimizations
        logger.info("Applying M2 optimizations")
        M2Optimizer.optimize_memory_usage()

        # Initialize components
        logger.info("Initializing pipeline components")
        self.preprocessor = DataPreprocessor(self.input_dir)
        self.nlp_model = BiomedicalNLP()
        self.extractor = AssociationExtractor(self.nlp_model)
        self.visualizer = ResultsVisualizer()

    def process_documents(self) -> pd.DataFrame:
        """Process all documents in the input directory."""
        logger.info(f"Processing documents from {self.input_dir}")
        documents = self.preprocessor.load_text_files()

        if not documents:
            logger.warning("No documents found in input directory")
            return pd.DataFrame()

        logger.info(f"Found {len(documents)} documents")
        processed_data = self.preprocessor.process_documents(documents)

        # Save processed data
        output_path = self.processed_dir / "processed_documents.csv"
        processed_data.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

        return processed_data

    def extract_associations(self, processed_data: pd.DataFrame) -> list:
        """Extract gene-disease associations from processed data."""
        logger.info("Extracting gene-disease associations")
        associations = self.extractor.extract_associations(
            processed_data["sentence"].tolist()
        )
        logger.info(f"Found {len(associations)} associations")
        return associations

    def generate_outputs(self, associations: list):
        """Generate visualizations and reports."""
        logger.info("Generating outputs")

        # Create network visualization
        network_graph = self.visualizer.create_network_graph(associations)
        network_path = self.generated_dir / "association_network.html"
        network_graph.write_html(str(network_path))
        logger.info(f"Saved network visualization to {network_path}")

        # Generate report
        report_path = self.reports_dir / "association_report.csv"
        self.visualizer.generate_report(associations, str(report_path))
        logger.info(f"Saved association report to {report_path}")

        # Get network analysis
        analysis = self.extractor.get_network_analysis()
        analysis_path = self.reports_dir / "network_analysis.txt"
        with open(analysis_path, "w") as f:
            for key, value in analysis.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Saved network analysis to {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Literature Mining Tool for Gene-Disease Associations"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing input documents (default: data/raw)",
        default=str(RAW_DATA_DIR),
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Initialize pipeline
        pipeline = LiteratureMining(Path(args.input_dir))

        # Process documents
        processed_data = pipeline.process_documents()
        if processed_data.empty:
            logger.error("No data to process. Exiting.")
            sys.exit(1)

        # Extract associations
        associations = pipeline.extract_associations(processed_data)
        if not associations:
            logger.warning("No associations found")
            sys.exit(0)

        # Generate outputs
        pipeline.generate_outputs(associations)

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.exception("An error occurred during processing")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import logging
import sys
import time
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_data_cleaning():
    """Run data cleaning step by executing the data_cleaning module"""
    logger.info("=" * 50)
    logger.info("STEP 1: DATA CLEANING")
    logger.info("=" * 50)
    
    try:
        logger.info("Running data cleaning...")
        from src.data_cleaning import clean_data
        clean_data()
        logger.info("✅ Data cleaning completed")
        return True
    except Exception as e:
        logger.error(f"❌ Data cleaning failed: {str(e)}")
        return False

def run_data_splitting():
    """Run data splitting step by calling split_data main function"""
    logger.info("=" * 50)
    logger.info("STEP 2: DATA SPLITTING")
    logger.info("=" * 50)
    
    try:
        logger.info("Running data splitting...")
        # Import and call the main block from split_data
        from src.split_data import split_data
        split_data()
        
        logger.info("✅ Data splitting completed")
        return True
    except Exception as e:
        logger.error(f"❌ Data splitting failed: {str(e)}", exc_info=True)
        return False

def run_feature_selection():
    """Run feature selection step by calling feature_selection main function"""
    logger.info("=" * 50)
    logger.info("STEP 3: FEATURE SELECTION")
    logger.info("=" * 50)
    
    try:
        logger.info("Running feature selection...")
        # Import and call the main function from feature_selection
        from feature_selection import run_feature_selection
        
        output_filenames = []

        output_filename = run_feature_selection(
            method="rfecv",
            output_dir="feature_sets",
            feature_subset="base",
            apply_scale_transform=False,
            apply_pca_img_transform=False,
            n_pca_components=None,
            rfe_step_size=None,
            include_location_features=True,
        )
        output_filenames.append(output_filename)

        output_filename = run_feature_selection(
            method="rfecv",
            output_dir="feature_sets",
            feature_subset="all",
            apply_scale_transform=False,
            apply_pca_img_transform=True,
            n_pca_components=0.8,
            rfe_step_size=10,
            include_location_features=True,
        )
        output_filenames.append(output_filename)

        output_filename = run_feature_selection(
            method="rfecv",
            output_dir="feature_sets",
            feature_subset="all",
            apply_scale_transform=False,
            apply_pca_img_transform=False,
            n_pca_components=None,
            rfe_step_size=10,
            include_location_features=True,
        )
        output_filenames.append(output_filename)

        output_filename = run_feature_selection(
            method="rfecv",
            output_dir="feature_sets",
            feature_subset="base_poi_pano",
            apply_scale_transform=False,
            apply_pca_img_transform=False,
            n_pca_components=None,
            rfe_step_size=10,
            include_location_features=True,
        )
        
        output_filenames.append(output_filename)


        
        logger.info("✅ Feature selection completed")
        return output_filenames
    except Exception as e:
        logger.error(f"❌ Feature selection failed: {str(e)}", exc_info=True)
        return False

def run_experiments(output_filenames):
    """Run experiments step by calling run_experiment main function"""
    logger.info("=" * 50)
    logger.info("STEP 4: RUNNING EXPERIMENTS")
    logger.info("=" * 50)
    
    try:
        logger.info("Running experiments...")
        # Import and call the main function from run_experiment
        from src.run_experiment import run_experiment_main
        
        run_experiment_main(feature_sets=output_filenames)
        
        logger.info("✅ Experiments completed")
        return True
    except Exception as e:
        logger.error(f"❌ Experiments failed: {str(e)}")
        return False

def main():
    """Main pipeline orchestrator"""
    parser = argparse.ArgumentParser(
        description="Run the complete ML pipeline for real estate price prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Pipeline steps to run (comma-separated): 'clean', 'split', 'feature', 'experiment', or 'all' (default: all)"
    )
    
    args = parser.parse_args()
    
    # Parse steps
    if args.steps.lower() == "all":
        steps_to_run = ["clean", "split", "feature", "experiment"]
    else:
        steps_to_run = [step.strip().lower() for step in args.steps.split(",")]
    
    # Validate steps
    valid_steps = ["clean", "split", "feature", "experiment"]
    invalid_steps = [step for step in steps_to_run if step not in valid_steps]
    if invalid_steps:
        logger.error(f"❌ Invalid steps: {invalid_steps}. Valid steps are: {valid_steps}")
        sys.exit(1)
    
    logger.info("🚀 Starting ML Pipeline")
    logger.info(f"Steps to run: {steps_to_run}")
    
    start_time = time.time()
    failed_steps = []
    
    # Execute pipeline steps
    if "clean" in steps_to_run:
        if not run_data_cleaning():
            failed_steps.append("clean")
    
    if "split" in steps_to_run and "clean" not in failed_steps:
        if not run_data_splitting():
            failed_steps.append("split")
    
    output_filenames = None
    if "feature" in steps_to_run and "split" not in failed_steps:
        output_filenames = run_feature_selection()
        if not output_filenames:
            failed_steps.append("feature")
    
    if "experiment" in steps_to_run and "feature" not in failed_steps:
        if not run_experiments(output_filenames):
            failed_steps.append("experiment")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info("=" * 50)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"⏱️  Total time: {duration:.1f} seconds")
    
    if failed_steps:
        logger.error(f"❌ Failed steps: {failed_steps}")
        sys.exit(1)
    else:
        logger.info("🎉 Pipeline completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Summary script to extract and analyze the best performing models from MLflow experiments.

This script queries all MLflow experiments, finds the best performing runs based on 
specified metrics, and creates comprehensive summaries in multiple formats.
"""

import json
import logging
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MLflowSummarizer:
    """Class to summarize MLflow experiment results."""
    
    def __init__(self, mlflow_uri: str = None):
        """
        Initialize the summarizer.
        
        Args:
            mlflow_uri: MLflow tracking URI. If None, uses default local tracking.
        """
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        
        self.client = MlflowClient()
        self.experiments = {}
        self.all_runs = []
        self.best_runs = {}
        
    def get_all_experiments(self) -> List[Dict]:
        """Get all MLflow experiments."""
        try:
            experiments = self.client.search_experiments()
            self.experiments = {exp.experiment_id: exp for exp in experiments}
            logger.info(f"Found {len(experiments)} experiments")
            return experiments
        except Exception as e:
            logger.error(f"Error fetching experiments: {e}")
            return []
    
    def get_runs_for_experiment(self, experiment_id: str) -> List[Dict]:
        """Get all runs for a specific experiment that are tagged as 'best_run'."""
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="tags.run_type = 'best_run'",  # Only get best runs
                max_results=1000  # Adjust if you have more runs
            )
            return runs
        except Exception as e:
            logger.error(f"Error fetching runs for experiment {experiment_id}: {e}")
            return []
    
    def extract_run_info(self, run) -> Dict[str, Any]:
        """Extract relevant information from a run."""
        try:
            run_info = {
                'experiment_id': run.info.experiment_id,
                'run_id': run.info.run_id,
                'run_name': run.info.run_name,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                'end_time': datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                'status': run.info.status,
                'experiment_name': self.experiments.get(run.info.experiment_id, {}).name if hasattr(self.experiments.get(run.info.experiment_id, {}), 'name') else 'Unknown'
            }
            
            # Extract tags
            tags = run.data.tags
            run_info['run_type'] = tags.get('run_type', 'Unknown')
            run_info['feature_set_tag'] = tags.get('feature_set', 'Unknown')
            
            # Extract parameters
            params = run.data.params
            run_info.update({
                'model_name': self._extract_model_name(run),
                'feature_set': params.get('feature_set', params.get('feature_set_file', run_info['feature_set_tag'])),
                'cv_folds': params.get('cv_folds', 'Unknown'),
                'random_state': params.get('random_state', 'Unknown'),
                'n_trials_optuna': params.get('n_trials_optuna', 'Unknown'),
                'tuning_scoring_metric': params.get('tuning_scoring_metric', 'Unknown')
            })
            
            # Extract metrics
            metrics = run.data.metrics
            
            # Log available metrics for debugging
            if metrics:
                logger.debug(f"Run {run.info.run_id} has metrics: {list(metrics.keys())}")
            
            # Common metric patterns to look for
            metric_patterns = {
                'train_score': ['train_final_mape', 'cv_mape'],
                'test_score': ['test_final_mape', 'best_optuna_cv_mape'],
                'train_mape': ['train_final_mape'],
                'test_mape': ['test_final_mape', 'best_optuna_cv_mape'],
                'train_rmse': ['train_final_rmse'],
                'test_rmse': ['test_final_rmse'],
                'train_r2': ['train_final_r2'],
                'test_r2': ['test_final_r2'],
                'cv_mape': ['cv_mape', 'best_optuna_cv_mape'],
                'tuning_duration': ['tuning_duration_sec'],
                'total_duration': ['total_pipeline_duration_sec']
            }
            
            # Extract metrics based on patterns
            for metric_key, patterns in metric_patterns.items():
                for pattern in patterns:
                    if pattern in metrics:
                        run_info[metric_key] = metrics[pattern]
                        break
                else:
                    run_info[metric_key] = None
            
            # Add all available metrics for completeness
            run_info['all_metrics'] = dict(metrics)
            run_info['all_params'] = dict(params)
            run_info['all_tags'] = dict(tags)
            
            return run_info
            
        except Exception as e:
            logger.error(f"Error extracting run info: {e}")
            return {}
    
    def _extract_model_name(self, run) -> str:
        """Extract model name from run data."""
        # Try multiple sources for model name
        params = run.data.params
        
        # Check common parameter names
        model_name_keys = ['model_name', 'model', 'algorithm']
        for key in model_name_keys:
            if key in params:
                return params[key]
        
        # Check run name for model info
        run_name = run.info.run_name or ''
        common_models = ['LightGBM', 'XGBoost', 'RandomForest', 'SVM', 'LinearRegression']
        for model in common_models:
            if model.lower() in run_name.lower():
                return model
        
        # Check tags
        tags = run.data.tags
        if 'model_name' in tags:
            return tags['model_name']
        
        return 'Unknown'
    
    def collect_all_runs(self) -> List[Dict]:
        """Collect all runs from all experiments."""
        experiments = self.get_all_experiments()
        all_runs = []
        total_runs = 0
        best_runs_count = 0
        
        for experiment in experiments:
            logger.info(f"Processing experiment: {experiment.name}")
            runs = self.get_runs_for_experiment(experiment.experiment_id)
            total_runs += len(runs)
            
            for run in runs:
                run_info = self.extract_run_info(run)
                if run_info and run_info.get('status') == 'FINISHED':
                    all_runs.append(run_info)
                    if run_info.get('run_type') == 'best_run':
                        best_runs_count += 1
        
        self.all_runs = all_runs
        logger.info(f"Collected {len(all_runs)} completed runs out of {total_runs} total runs")
        logger.info(f"Found {best_runs_count} runs tagged as 'best_run'")
        
        # Debug: Show available metrics from a sample of runs
        if all_runs:
            sample_run = all_runs[0]
            logger.info(f"Sample run metrics: {list(sample_run.get('all_metrics', {}).keys())}")
            
            # Count runs with different metric types
            metric_counts = {}
            for run in all_runs:
                for metric in run.get('all_metrics', {}):
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1
            
            logger.info("Metric availability across all runs:")
            for metric, count in sorted(metric_counts.items()):
                logger.info(f"  {metric}: {count} runs")
        
        return all_runs
    
    def collect_best_runs_by_tag(self) -> List[Dict]:
        """Collect runs that are tagged as 'best_run'."""
        if not self.all_runs:
            self.collect_all_runs()
        
        best_runs = [run for run in self.all_runs if run.get('run_type') == 'best_run']
        logger.info(f"Found {len(best_runs)} runs tagged as 'best_run'")
        
        if best_runs:
            # Group by model-feature combination for summary
            grouped_runs = {}
            for run in best_runs:
                key = f"{run['model_name']}_{run['feature_set']}"
                if key not in grouped_runs:
                    grouped_runs[key] = []
                grouped_runs[key].append(run)
            
            # Use the most recent run for each combination if there are duplicates
            final_best_runs = {}
            for key, runs in grouped_runs.items():
                if len(runs) > 1:
                    logger.warning(f"Multiple best runs found for {key}, using most recent")
                    most_recent = max(runs, key=lambda x: x['start_time'])
                    final_best_runs[key] = most_recent
                else:
                    final_best_runs[key] = runs[0]
            
            self.best_runs = final_best_runs
            logger.info(f"Selected {len(final_best_runs)} unique best runs")
        
        return best_runs
    
    def find_best_runs(self, metric: str = 'test_mape', maximize: bool = False) -> Dict[str, Dict]:
        """
        Find the best run for each model-feature_set combination.
        
        Args:
            metric: Metric to optimize for
            maximize: Whether to maximize (True) or minimize (False) the metric
        """
        if not self.all_runs:
            self.collect_all_runs()
        
        # Filter runs that have the specified metric
        valid_runs = [run for run in self.all_runs if run.get(metric) is not None]
        
        if not valid_runs:
            logger.warning(f"No runs found with metric '{metric}'")
            return {}
        
        logger.info(f"Finding best runs based on '{metric}' (maximize={maximize})")
        logger.info(f"Evaluating {len(valid_runs)} runs with valid '{metric}' values")
        
        # Group runs by model-feature_set combination
        grouped_runs = {}
        for run in valid_runs:
            key = f"{run['model_name']}_{run['feature_set']}"
            if key not in grouped_runs:
                grouped_runs[key] = []
            grouped_runs[key].append(run)
        
        # Find best run in each group
        best_runs = {}
        for key, runs in grouped_runs.items():
            if maximize:
                best_run = max(runs, key=lambda x: x[metric])
            else:
                best_run = min(runs, key=lambda x: x[metric])
            
            best_runs[key] = best_run
        
        self.best_runs = best_runs
        logger.info(f"Found {len(best_runs)} best runs across {len(grouped_runs)} model-feature combinations")
        return best_runs
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame of the best runs."""
        if not self.best_runs:
            logger.warning("No best runs found. Call find_best_runs() first.")
            return pd.DataFrame()
        
        # Prepare data for DataFrame
        summary_data = []
        for key, run in self.best_runs.items():
            summary_row = {
                'model_feature_combination': key,
                'model_name': run['model_name'],
                'feature_set': run['feature_set'],
                'run_type': run.get('run_type'),
                'experiment_name': run['experiment_name'],
                'run_id': run['run_id'],
                'run_name': run['run_name'],
                'train_score': run.get('train_score'),
                'test_score': run.get('test_score'),
                'train_mape': run.get('train_mape'),
                'test_mape': run.get('test_mape'),
                'train_rmse': run.get('train_rmse'),
                'test_rmse': run.get('test_rmse'),
                'train_r2': run.get('train_r2'),
                'test_r2': run.get('test_r2'),
                'cv_mape': run.get('cv_mape'),
                'tuning_duration': run.get('tuning_duration'),
                'total_duration': run.get('total_duration'),
                'cv_folds': run.get('cv_folds'),
                'n_trials_optuna': run.get('n_trials_optuna'),
                'tuning_scoring_metric': run.get('tuning_scoring_metric'),
                'start_time': run['start_time'],
                'end_time': run['end_time']
            }
            summary_data.append(summary_row)
        
        df = pd.DataFrame(summary_data)
        
        # Sort by test_score if available, otherwise by train_score
        sort_column = 'test_score' if 'test_score' in df.columns and df['test_score'].notna().any() else 'train_score'
        if sort_column in df.columns:
            df = df.sort_values(sort_column, ascending=False, na_position='last')
        
        return df
    
    def export_summary(self, output_dir: str = ".", formats: List[str] = None) -> Dict[str, str]:
        """
        Export summary in multiple formats.
        
        Args:
            output_dir: Directory to save files
            formats: List of formats to export ('csv', 'json', 'excel')
        """
        if formats is None:
            formats = ['csv', 'json']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary DataFrame
        df = self.create_summary_dataframe()
        
        if df.empty:
            logger.warning("No data to export")
            return {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}
        
        # Export CSV
        if 'csv' in formats:
            csv_path = os.path.join(output_dir, f"experiment_summary_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            exported_files['csv'] = csv_path
            logger.info(f"Summary exported to CSV: {csv_path}")
        
        # Export JSON
        if 'json' in formats:
            json_path = os.path.join(output_dir, f"best_models_report_{timestamp}.json")
            
            # Create detailed JSON report with comprehensive run information
            best_runs_detailed = {}
            for key, run in self.best_runs.items():
                best_runs_detailed[key] = {
                    'basic_info': {
                        'run_id': run['run_id'],
                        'run_name': run['run_name'],
                        'experiment_name': run['experiment_name'],
                        'model_name': run['model_name'],
                        'feature_set': run['feature_set'],
                        'run_type': run.get('run_type'),
                        'start_time': run['start_time'].isoformat() if run['start_time'] else None,
                        'end_time': run['end_time'].isoformat() if run['end_time'] else None,
                        'status': run['status']
                    },
                    'performance_metrics': {
                        'train_score': run.get('train_score'),
                        'test_score': run.get('test_score'),
                        'train_mape': run.get('train_mape'),
                        'test_mape': run.get('test_mape'),
                        'train_rmse': run.get('train_rmse'),
                        'test_rmse': run.get('test_rmse'),
                        'train_r2': run.get('train_r2'),
                        'test_r2': run.get('test_r2'),
                        'cv_mape': run.get('cv_mape'),
                        'tuning_duration': run.get('tuning_duration'),
                        'total_duration': run.get('total_duration')
                    },
                    'hyperparameters': run.get('all_params', {}),
                    'tags': run.get('all_tags', {}),
                    'all_metrics': run.get('all_metrics', {}),
                    'training_config': {
                        'cv_folds': run.get('cv_folds'),
                        'random_state': run.get('random_state'),
                        'n_trials_optuna': run.get('n_trials_optuna'),
                        'tuning_scoring_metric': run.get('tuning_scoring_metric')
                    }
                }
            
            report = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_experiments': len(self.experiments),
                    'total_runs_analyzed': len(self.all_runs),
                    'best_runs_found': len(self.best_runs),
                    'optimization_metric': getattr(self, '_optimization_metric', 'test_score'),
                    'description': 'Comprehensive report of best performing models with all hyperparameters, tags, and metrics'
                },
                'summary_statistics': {
                    'unique_models': df['model_name'].nunique() if not df.empty else 0,
                    'unique_feature_sets': df['feature_set'].nunique() if not df.empty else 0,
                    'unique_experiments': df['experiment_name'].nunique() if not df.empty else 0
                },
                'best_runs_detailed': best_runs_detailed,
                'summary_table': df.to_dict('records') if not df.empty else []
            }
            
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            exported_files['json'] = json_path
            logger.info(f"Detailed report exported to JSON: {json_path}")
        
        # Export Excel
        if 'excel' in formats:
            excel_path = os.path.join(output_dir, f"experiment_summary_{timestamp}.xlsx")
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Best_Runs', index=False)
                    
                    # Create additional sheets with analysis
                    if not df.empty:
                        # Model comparison
                        model_summary = df.groupby('model_name').agg({
                            'test_score': ['count', 'mean', 'std', 'max', 'min'],
                            'test_mape': ['mean', 'std', 'min'] if 'test_mape' in df.columns else None
                        }).round(4)
                        model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns]
                        model_summary.to_excel(writer, sheet_name='Model_Comparison')
                        
                        # Feature set comparison
                        feature_summary = df.groupby('feature_set').agg({
                            'test_score': ['count', 'mean', 'std', 'max', 'min'],
                            'test_mape': ['mean', 'std', 'min'] if 'test_mape' in df.columns else None
                        }).round(4)
                        feature_summary.columns = ['_'.join(col).strip() for col in feature_summary.columns]
                        feature_summary.to_excel(writer, sheet_name='FeatureSet_Comparison')
                
                exported_files['excel'] = excel_path
                logger.info(f"Summary exported to Excel: {excel_path}")
            except ImportError:
                logger.warning("openpyxl not installed. Skipping Excel export.")
        
        return exported_files
    
    def print_summary(self, top_n: int = 10):
        """Print a summary of the best runs to console."""
        if not self.best_runs:
            logger.warning("No best runs to display. Call find_best_runs() first.")
            return
        
        df = self.create_summary_dataframe()
        
        if df.empty:
            print("No data to display")
            return
        
        print("\n" + "="*80)
        print(f"EXPERIMENT SUMMARY - TOP {min(top_n, len(df))} BEST RUNS")
        print("="*80)
        
        # Display summary statistics
        print(f"Total experiments analyzed: {len(self.experiments)}")
        print(f"Total completed runs: {len(self.all_runs)}")
        print(f"Best runs found: {len(self.best_runs)}")
        print(f"Unique models: {df['model_name'].nunique()}")
        print(f"Unique feature sets: {df['feature_set'].nunique()}")
        
        print("\n" + "-"*80)
        print("TOP PERFORMING RUNS:")
        print("-"*80)
        
        # Display top runs
        display_columns = ['model_name', 'feature_set', 'test_score', 'train_score', 'test_mape', 'test_rmse']
        available_columns = [col for col in display_columns if col in df.columns and df[col].notna().any()]
        
        top_runs = df.head(top_n)[available_columns]
        print(top_runs.to_string(index=False, float_format='%.4f'))
        
        # Display model performance summary
        if 'test_score' in df.columns:
            print("\n" + "-"*80)
            print("MODEL PERFORMANCE SUMMARY:")
            print("-"*80)
            model_stats = df.groupby('model_name')['test_score'].agg(['count', 'mean', 'std', 'max']).round(4)
            model_stats.columns = ['Runs', 'Mean_Score', 'Std_Score', 'Best_Score']
            print(model_stats.to_string())


def main():
    """Main function to run the summarization."""
    print("MLflow Experiment Summarizer")
    print("="*50)
    
    # Initialize summarizer
    summarizer = MLflowSummarizer()
    
    # Collect runs tagged as 'best_run'
    logger.info("Collecting runs tagged as 'best_run'...")
    summarizer.collect_all_runs()
    
    if not summarizer.all_runs:
        logger.error("No runs tagged as 'best_run' found in MLflow")
        return
    
    # Use the collected best runs directly for the summary
    grouped_runs = {}
    for run in summarizer.all_runs:
        key = f"{run['model_name']}_{run['feature_set']}"
        if key not in grouped_runs:
            grouped_runs[key] = []
        grouped_runs[key].append(run)
    
    # Use the most recent run for each combination if there are duplicates
    final_best_runs = {}
    for key, runs in grouped_runs.items():
        if len(runs) > 1:
            logger.warning(f"Multiple best runs found for {key}, using most recent")
            most_recent = max(runs, key=lambda x: x['start_time'])
            final_best_runs[key] = most_recent
        else:
            final_best_runs[key] = runs[0]
    
    summarizer.best_runs = final_best_runs
    logger.info(f"Selected {len(final_best_runs)} unique best runs")
    
    # Print summary to console
    summarizer.print_summary(top_n=15)
    
    # Export summary files
    logger.info("Exporting summary files...")
    exported_files = summarizer.export_summary(
        output_dir="experiment_summaries",
        formats=['csv', 'json', 'excel']
    )
    
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    for format_type, file_path in exported_files.items():
        print(f"{format_type.upper()}: {file_path}")
    
    print(f"\nSummary complete! Check the 'experiment_summaries' directory for detailed reports.")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Compare Kronos and Chronos 24h predictions

This script loads the JSON outputs from both models and provides
a comparative analysis of their predictions.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta

# Paths
project_root = Path(__file__).parent.parent.parent
results_dir = project_root / 'experiments' / 'zero_shot'
plots_dir = project_root / 'evaluation' / 'plots'

# Ensure plots directory exists
plots_dir.mkdir(parents=True, exist_ok=True)

def load_predictions():
    """Load prediction results from both models"""
    
    kronos_path = results_dir / 'kronos_zero_shot_24h_prediction.json'
    chronos_path = results_dir / 'chronos_zero_shot_24h_prediction.json'
    
    results = {}
    
    if kronos_path.exists():
        with open(kronos_path, 'r') as f:
            results['kronos'] = json.load(f)
        print(f"✓ Loaded Kronos predictions from {kronos_path}")
    else:
        print(f"⚠ Kronos predictions not found at {kronos_path}")
        
    if chronos_path.exists():
        with open(chronos_path, 'r') as f:
            results['chronos'] = json.load(f)
        print(f"✓ Loaded Chronos predictions from {chronos_path}")
    else:
        print(f"⚠ Chronos predictions not found at {chronos_path}")
    
    return results

def extract_close_prices(results):
    """Extract close prices and actual values for comparison"""
    
    comparison_data = {}
    
    for model_name, data in results.items():
        timestamps = []
        predicted_prices = []
        actual_prices = []
        confidence_bounds = {'lower': [], 'upper': []} if model_name == 'chronos' else None
        
        for pred in data['predictions']:
            timestamps.append(pred['timestamp'])
            
            if model_name == 'kronos':
                # Kronos has OHLC prices and actual values
                predicted_prices.append(pred['predicted_prices']['close'])
                actual_prices.append(pred['actual_prices']['close'])
            else:
                # Chronos has median price, actual values and confidence intervals
                predicted_prices.append(pred['price_median'])
                actual_prices.append(pred['actual_price'])
                confidence_bounds['lower'].append(pred['price_q10'])
                confidence_bounds['upper'].append(pred['price_q90'])
        
        comparison_data[model_name] = {
            'timestamps': timestamps,
            'predicted_prices': predicted_prices,
            'actual_prices': actual_prices,
            'confidence_bounds': confidence_bounds
        }
    
    return comparison_data


def create_comparison_plot(comparison_data, results):
    """Create a visual comparison plot of both model predictions vs actual values"""
    
    if len(comparison_data) == 0:
        print("No data to plot")
        return None
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Get actual values (should be same for both models)
    actual_values = None
    timestamps = None
    
    # Plot 1: Actual vs Predicted for each model
    for model_name, data in comparison_data.items():
        timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
        predicted_prices = data['predicted_prices']
        actual_prices = data['actual_prices']
        
        if actual_values is None:
            actual_values = actual_prices
        
        if model_name == 'kronos':
            ax1.plot(timestamps, predicted_prices, 'o-', label=f'Kronos Predicted',
                    color='blue', linewidth=2, markersize=6)
        elif model_name == 'chronos':
            ax1.plot(timestamps, predicted_prices, 's-', label=f'Chronos Predicted',
                    color='red', linewidth=2, markersize=6)
            
            # Add confidence interval for Chronos
            if data['confidence_bounds']:
                lower_bounds = data['confidence_bounds']['lower']
                upper_bounds = data['confidence_bounds']['upper']
                ax1.fill_between(timestamps, lower_bounds, upper_bounds,
                               alpha=0.2, color='red', label='Chronos 80% Confidence Interval')
    
    # Plot actual values
    if actual_values and timestamps:
        ax1.plot(timestamps, actual_values, 'k*-', label='Actual',
                color='black', linewidth=3, markersize=8)
    
    # Formatting for first subplot
    ax1.set_title('24-Hour Energy Price Forecast: Actual vs Predicted',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Price (€/MWh)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Model comparison (predicted only)
    for model_name, data in comparison_data.items():
        timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
        predicted_prices = data['predicted_prices']
        
        if model_name == 'kronos':
            ax2.plot(timestamps, predicted_prices, 'o-', label=f'Kronos',
                    color='blue', linewidth=2, markersize=6)
        elif model_name == 'chronos':
            ax2.plot(timestamps, predicted_prices, 's-', label=f'Chronos',
                    color='red', linewidth=2, markersize=6)
    
    # Formatting for second subplot
    ax2.set_title('Model Predictions Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Price (€/MWh)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add performance statistics as text
    if len(comparison_data) == 2:
        kronos_predicted = comparison_data['kronos']['predicted_prices']
        chronos_predicted = comparison_data['chronos']['predicted_prices']
        
        # Model comparison stats
        correlation = np.corrcoef(kronos_predicted, chronos_predicted)[0, 1]
        mae_models = np.mean(np.abs(np.array(kronos_predicted) - np.array(chronos_predicted)))
        
        # Get performance metrics from results
        kronos_mae = results['kronos']['performance_metrics']['close']['mae']
        chronos_mae = results['chronos']['performance_metrics']['mae']
        
        # Add text box with statistics
        textstr = f'Model Correlation: {correlation:.3f}\nMAE between models: {mae_models:.2f} €/MWh\nKronos MAE: {kronos_mae:.2f} €/MWh\nChronos MAE: {chronos_mae:.2f} €/MWh'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = plots_dir / f'actual_vs_predicted_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"✓ Comparison plot saved: {plot_path}")
    
    # Show the plot
    plt.show()
    
    return plot_path

def calculate_comparison_metrics(comparison_data):
    """Calculate comparison metrics between models and vs actual values"""
    
    if len(comparison_data) < 2:
        print("Need both models for comparison")
        return None
    
    kronos_predicted = np.array(comparison_data['kronos']['predicted_prices'])
    chronos_predicted = np.array(comparison_data['chronos']['predicted_prices'])
    actual_values = np.array(comparison_data['kronos']['actual_prices'])  # Same for both models
    
    # Basic statistics for predictions
    metrics = {
        'kronos_predicted_stats': {
            'mean': float(kronos_predicted.mean()),
            'std': float(kronos_predicted.std()),
            'min': float(kronos_predicted.min()),
            'max': float(kronos_predicted.max())
        },
        'chronos_predicted_stats': {
            'mean': float(chronos_predicted.mean()),
            'std': float(chronos_predicted.std()),
            'min': float(chronos_predicted.min()),
            'max': float(chronos_predicted.max())
        },
        'actual_stats': {
            'mean': float(actual_values.mean()),
            'std': float(actual_values.std()),
            'min': float(actual_values.min()),
            'max': float(actual_values.max())
        },
        'model_comparison': {
            'predicted_mean_difference': float(kronos_predicted.mean() - chronos_predicted.mean()),
            'predicted_correlation': float(np.corrcoef(kronos_predicted, chronos_predicted)[0, 1]),
            'mae_between_predictions': float(np.mean(np.abs(kronos_predicted - chronos_predicted))),
            'rmse_between_predictions': float(np.sqrt(np.mean((kronos_predicted - chronos_predicted) ** 2)))
        },
        'accuracy_comparison': {
            'kronos_vs_actual': {
                'mae': float(np.mean(np.abs(kronos_predicted - actual_values))),
                'rmse': float(np.sqrt(np.mean((kronos_predicted - actual_values) ** 2))),
                'correlation': float(np.corrcoef(kronos_predicted, actual_values)[0, 1])
            },
            'chronos_vs_actual': {
                'mae': float(np.mean(np.abs(chronos_predicted - actual_values))),
                'rmse': float(np.sqrt(np.mean((chronos_predicted - actual_values) ** 2))),
                'correlation': float(np.corrcoef(chronos_predicted, actual_values)[0, 1])
            }
        }
    }
    
    return metrics

def generate_comparison_report(results, comparison_data, metrics):
    """Generate comprehensive comparison report"""
    
    report = {
        'comparison_info': {
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(results.keys())
        },
        'model_details': {},
        'forecast_comparison': {
            'forecast_period': None,
            'prediction_length': None
        },
        'price_comparison': metrics,
        'detailed_hourly_comparison': []
    }
    
    # Extract model details
    for model_name, data in results.items():
        report['model_details'][model_name] = {
            'model_info': data['model_info'],
            'data_source': data['data_source'],
            'model_specific': data.get('model_specific', {})
        }
    
    # Set forecast period (should be same for both)
    if results:
        first_model = list(results.values())[0]
        report['forecast_comparison']['forecast_period'] = first_model['forecast_info']['period']
        report['forecast_comparison']['prediction_length'] = first_model['forecast_info']['prediction_length']
    
    # Detailed hourly comparison
    if len(comparison_data) == 2:
        kronos_data = comparison_data['kronos']
        chronos_data = comparison_data['chronos']
        
        for i in range(len(kronos_data['timestamps'])):
            hour_comparison = {
                'hour': i + 1,
                'timestamp': kronos_data['timestamps'][i],
                'actual_price': kronos_data['actual_prices'][i],
                'kronos_predicted': kronos_data['predicted_prices'][i],
                'chronos_predicted': chronos_data['predicted_prices'][i],
                'kronos_error': kronos_data['predicted_prices'][i] - kronos_data['actual_prices'][i],
                'chronos_error': chronos_data['predicted_prices'][i] - chronos_data['actual_prices'][i],
                'prediction_difference': kronos_data['predicted_prices'][i] - chronos_data['predicted_prices'][i],
                'kronos_abs_error': abs(kronos_data['predicted_prices'][i] - kronos_data['actual_prices'][i]),
                'chronos_abs_error': abs(chronos_data['predicted_prices'][i] - chronos_data['actual_prices'][i])
            }
            report['detailed_hourly_comparison'].append(hour_comparison)
    
    return report

def main():
    """Main execution function"""
    print("=" * 60)
    print("KRONOS vs CHRONOS PREDICTION COMPARISON")
    print("=" * 60)
    
    # Load predictions
    results = load_predictions()
    
    if len(results) == 0:
        print("No prediction files found. Run the individual prediction scripts first.")
        return 1
    
    # Extract comparable price data
    comparison_data = extract_close_prices(results)
    
    # Create visual comparison plot
    print("\nCreating comparison visualization...")
    plot_path = create_comparison_plot(comparison_data, results)
    
    # Calculate comparison metrics
    metrics = calculate_comparison_metrics(comparison_data)
    
    # Generate comprehensive report
    report = generate_comparison_report(results, comparison_data, metrics)
    
    # Add plot information to report
    if plot_path:
        report['visualization'] = {
            'plot_path': str(plot_path),
            'plot_timestamp': datetime.now().isoformat()
        }
    
    # Display summary
    print(f"\n✓ Comparison completed for {len(results)} model(s)")
    
    if metrics:
        print("\nActual vs Predicted Performance:")
        print(f"  Actual average: {metrics['actual_stats']['mean']:.3f} €/MWh")
        print(f"  Kronos predicted average: {metrics['kronos_predicted_stats']['mean']:.3f} €/MWh")
        print(f"  Chronos predicted average: {metrics['chronos_predicted_stats']['mean']:.3f} €/MWh")
        
        print(f"\nAccuracy Comparison:")
        print(f"  Kronos MAE: {metrics['accuracy_comparison']['kronos_vs_actual']['mae']:.3f} €/MWh")
        print(f"  Chronos MAE: {metrics['accuracy_comparison']['chronos_vs_actual']['mae']:.3f} €/MWh")
        print(f"  Kronos RMSE: {metrics['accuracy_comparison']['kronos_vs_actual']['rmse']:.3f} €/MWh")
        print(f"  Chronos RMSE: {metrics['accuracy_comparison']['chronos_vs_actual']['rmse']:.3f} €/MWh")
        
        print(f"\nModel Correlation with Actual:")
        print(f"  Kronos: {metrics['accuracy_comparison']['kronos_vs_actual']['correlation']:.3f}")
        print(f"  Chronos: {metrics['accuracy_comparison']['chronos_vs_actual']['correlation']:.3f}")
        
        print(f"\nModel Comparison:")
        print(f"  Prediction correlation: {metrics['model_comparison']['predicted_correlation']:.3f}")
        print(f"  MAE between predictions: {metrics['model_comparison']['mae_between_predictions']:.3f} €/MWh")
        
        # Determine which model performed better
        kronos_mae = metrics['accuracy_comparison']['kronos_vs_actual']['mae']
        chronos_mae = metrics['accuracy_comparison']['chronos_vs_actual']['mae']
        better_model = "Chronos" if chronos_mae < kronos_mae else "Kronos"
        print(f"\n🏆 Better performing model: {better_model} (lower MAE)")
    
    # Save comparison report
    report_path = results_dir / 'model_comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Detailed comparison report saved: {report_path}")
    if plot_path:
        print(f"✓ Comparison plot saved: {plot_path}")
    
    # Also output JSON to console (abbreviated version without detailed hourly data)
    report_summary = {k: v for k, v in report.items() if k != 'detailed_hourly_comparison'}
    print("\nComparison Report Summary:")
    print(json.dumps(report_summary, indent=2))
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETED")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
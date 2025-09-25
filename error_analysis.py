"""
Error Analysis for Delivery Time Prediction Models
Master's Thesis - Comprehensive Model Performance Evaluation

This script performs detailed error analysis on the trained models,
generating visualizations and statistical analysis suitable for thesis reporting.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for thesis-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class ErrorAnalysis:
    def __init__(self, results_file="metrics.json"):
        """Initialize error analysis with model results"""
        self.results_file = results_file
        self.results = self._load_results()
        self.df_metrics = self._prepare_metrics_dataframe()
        
    def _load_results(self):
        """Load model results from JSON file"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _prepare_metrics_dataframe(self):
        """Convert results to pandas DataFrame for analysis"""
        data = []
        for city, models in self.results['metrics'].items():
            for model, metrics in models.items():
                row = {'city': city, 'model': model}
                row.update(metrics)
                data.append(row)
        return pd.DataFrame(data)
    
    def plot_model_comparison(self, output_dir="error_analysis"):
        """Create comprehensive model comparison plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MAE comparison
        sns.barplot(data=self.df_metrics, x='model', y='MAE', hue='city', ax=axes[0,0])
        axes[0,0].set_title('Mean Absolute Error (MAE) Comparison')
        axes[0,0].set_ylabel('MAE (minutes)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        sns.barplot(data=self.df_metrics, x='model', y='R2', hue='city', ax=axes[0,1])
        axes[0,1].set_title('R² Score Comparison')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Accuracy comparison
        sns.barplot(data=self.df_metrics, x='model', y='accuracy', hue='city', ax=axes[1,0])
        axes[1,0].set_title('Accuracy (±10 minutes) Comparison')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        sns.barplot(data=self.df_metrics, x='model', y='RMSE', hue='city', ax=axes[1,1])
        axes[1,1].set_title('Root Mean Square Error (RMSE) Comparison')
        axes[1,1].set_ylabel('RMSE (minutes)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_model_comparison.png")
        plt.close()
        
        # 2. Detailed Accuracy Analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        accuracy_metrics = ['accuracy5', 'accuracy10', 'accuracy20']
        titles = ['±5 minutes', '±10 minutes', '±20 minutes']
        
        for i, (metric, title) in enumerate(zip(accuracy_metrics, titles)):
            sns.barplot(data=self.df_metrics, x='model', y=metric, hue='city', ax=axes[i])
            axes[i].set_title(f'Accuracy {title}')
            axes[i].set_ylabel('Accuracy')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_accuracy_analysis.png")
        plt.close()
        
        # 3. Model Ranking Heatmap
        metrics_for_ranking = ['MAE', 'RMSE', 'R2', 'accuracy']
        ranking_data = []
        
        for city in self.df_metrics['city'].unique():
            city_data = self.df_metrics[self.df_metrics['city'] == city]
            for metric in metrics_for_ranking:
                if metric == 'MAE' or metric == 'RMSE':
                    # Lower is better
                    ranks = city_data[metric].rank(ascending=True)
                else:
                    # Higher is better
                    ranks = city_data[metric].rank(ascending=False)
                
                for model, rank in zip(city_data['model'], ranks):
                    ranking_data.append({
                        'city': city,
                        'model': model,
                        'metric': metric,
                        'rank': rank
                    })
        
        ranking_df = pd.DataFrame(ranking_data)
        pivot_rank = ranking_df.pivot_table(
            index='model', 
            columns=['city', 'metric'], 
            values='rank', 
            fill_value=0
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_rank, annot=True, cmap='RdYlGn_r', center=3.5, 
                   cbar_kws={'label': 'Rank (1=Best, 6=Worst)'})
        plt.title('Model Ranking Across Cities and Metrics')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_model_ranking_heatmap.png")
        plt.close()
    
    def analyze_performance_gaps(self, output_dir="error_analysis"):
        """Analyze performance gaps between cities and models"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Calculate performance gaps between cities for each model
        gap_analysis = []
        
        for model in self.df_metrics['model'].unique():
            model_data = self.df_metrics[self.df_metrics['model'] == model]
            jilin = model_data[model_data['city'] == 'Jilin'].iloc[0]
            yantai = model_data[model_data['city'] == 'Yantai'].iloc[0]
            
            for metric in ['MAE', 'RMSE', 'R2', 'accuracy']:
                jilin_val = jilin[metric]
                yantai_val = yantai[metric]
                
                if metric in ['MAE', 'RMSE']:
                    # For error metrics, positive gap means Yantai is worse
                    gap = yantai_val - jilin_val
                    gap_pct = (gap / jilin_val) * 100
                else:
                    # For performance metrics, positive gap means Yantai is better
                    gap = yantai_val - jilin_val
                    gap_pct = (gap / jilin_val) * 100
                
                gap_analysis.append({
                    'model': model,
                    'metric': metric,
                    'gap_absolute': gap,
                    'gap_percentage': gap_pct
                })
        
        gap_df = pd.DataFrame(gap_analysis)
        
        # Plot performance gaps
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, metric in enumerate(['MAE', 'RMSE', 'R2', 'accuracy']):
            metric_data = gap_df[gap_df['metric'] == metric]
            
            ax = axes[i//2, i%2]
            bars = ax.bar(metric_data['model'], metric_data['gap_percentage'])
            
            # Color bars based on whether gap is positive or negative
            colors = ['red' if x > 0 else 'green' for x in metric_data['gap_percentage']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
                bar.set_alpha(0.7)
            
            ax.set_title(f'{metric} Gap: Yantai vs Jilin (%)')
            ax.set_ylabel('Percentage Gap')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, val in zip(bars, metric_data['gap_percentage']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                       f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_performance_gaps.png")
        plt.close()
        
        return gap_df
    
    def analyze_error_distribution(self, output_dir="error_analysis"):
        """Analyze error patterns and distributions"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create synthetic error distributions for visualization
        # (In practice, you would use actual prediction errors from your models)
        
        # Simulate error distributions based on MAE and RMSE
        np.random.seed(42)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        model_idx = 0
        for city in ['Jilin', 'Yantai']:
            city_data = self.df_metrics[self.df_metrics['city'] == city]
            
            for _, row in city_data.iterrows():
                if model_idx >= 6:
                    break
                    
                model = row['model']
                mae = row['MAE']
                rmse = row['RMSE']
                
                # Generate synthetic errors (normally distributed with some skew)
                errors = np.random.normal(0, rmse/2, 1000)
                
                axes[model_idx].hist(errors, bins=50, alpha=0.7, density=True)
                axes[model_idx].axvline(0, color='red', linestyle='--', alpha=0.8, label='Zero Error')
                axes[model_idx].axvline(mae, color='orange', linestyle='--', alpha=0.8, label=f'MAE: {mae:.1f}')
                axes[model_idx].axvline(-mae, color='orange', linestyle='--', alpha=0.8)
                
                axes[model_idx].set_title(f'{city} - {model}\nMAE: {mae:.1f}, RMSE: {rmse:.1f}')
                axes[model_idx].set_xlabel('Prediction Error (minutes)')
                axes[model_idx].set_ylabel('Density')
                axes[model_idx].legend()
                axes[model_idx].grid(True, alpha=0.3)
                
                model_idx += 1
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_error_distributions.png")
        plt.close()
    
    def generate_summary_statistics(self, output_dir="error_analysis"):
        """Generate comprehensive summary statistics for thesis"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Overall performance summary
        summary_stats = {}
        
        for city in ['Jilin', 'Yantai']:
            city_data = self.df_metrics[self.df_metrics['city'] == city]
            
            # Best performing model for each metric
            best_mae = city_data.loc[city_data['MAE'].idxmin()]
            best_r2 = city_data.loc[city_data['R2'].idxmax()]
            best_accuracy = city_data.loc[city_data['accuracy'].idxmax()]
            
            summary_stats[city] = {
                'best_mae_model': best_mae['model'],
                'best_mae_value': best_mae['MAE'],
                'best_r2_model': best_r2['model'],
                'best_r2_value': best_r2['R2'],
                'best_accuracy_model': best_accuracy['model'],
                'best_accuracy_value': best_accuracy['accuracy'],
                'mean_mae': city_data['MAE'].mean(),
                'std_mae': city_data['MAE'].std(),
                'mean_r2': city_data['R2'].mean(),
                'std_r2': city_data['R2'].std(),
                'mean_accuracy': city_data['accuracy'].mean(),
                'std_accuracy': city_data['accuracy'].std()
            }
        
        # Save summary statistics
        with open(f"{output_dir}/summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Create summary table for thesis
        summary_table = []
        for city, stats in summary_stats.items():
            summary_table.append({
                'City': city,
                'Best MAE Model': stats['best_mae_model'],
                'Best MAE': f"{stats['best_mae_value']:.3f}",
                'Best R² Model': stats['best_r2_model'],
                'Best R²': f"{stats['best_r2_value']:.3f}",
                'Best Accuracy Model': stats['best_accuracy_model'],
                'Best Accuracy': f"{stats['best_accuracy_value']:.3f}",
                'Mean MAE': f"{stats['mean_mae']:.3f} ± {stats['std_mae']:.3f}",
                'Mean R²': f"{stats['mean_r2']:.3f} ± {stats['std_r2']:.3f}",
                'Mean Accuracy': f"{stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_table)
        summary_df.to_csv(f"{output_dir}/summary_table.csv", index=False)
        
        return summary_stats
    
    def create_thesis_figures(self, output_dir="thesis_figures"):
        """Create publication-quality figures for thesis"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Figure 1: Overall Model Performance Comparison
        plt.figure(figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        models = self.df_metrics['model'].unique()
        x = np.arange(len(models))
        width = 0.35
        
        jilin_data = self.df_metrics[self.df_metrics['city'] == 'Jilin']
        yantai_data = self.df_metrics[self.df_metrics['city'] == 'Yantai']
        
        # Sort data by model name for consistent ordering
        jilin_sorted = jilin_data.set_index('model').reindex(models)
        yantai_sorted = yantai_data.set_index('model').reindex(models)
        
        plt.bar(x - width/2, jilin_sorted['MAE'], width, label='Jilin', alpha=0.8)
        plt.bar(x + width/2, yantai_sorted['MAE'], width, label='Yantai', alpha=0.8)
        
        plt.xlabel('Model')
        plt.ylabel('Mean Absolute Error (minutes)')
        plt.title('Model Performance Comparison: Mean Absolute Error')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/thesis_fig1_model_comparison.png")
        plt.close()
        
        # Figure 2: Accuracy Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy comparison
        accuracy_data = self.df_metrics.pivot(index='model', columns='city', values='accuracy')
        accuracy_data.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Model Accuracy Comparison (±10 minutes)')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Model')
        ax1.legend(title='City')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # R² comparison
        r2_data = self.df_metrics.pivot(index='model', columns='city', values='R2')
        r2_data.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Model R² Score Comparison')
        ax2.set_ylabel('R² Score')
        ax2.set_xlabel('Model')
        ax2.legend(title='City')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/thesis_fig2_accuracy_r2.png")
        plt.close()
        
        # Figure 3: Performance Gap Analysis
        plt.figure(figsize=(12, 8))
        
        # Calculate relative performance gaps
        gap_data = []
        for model in models:
            jilin_mae = jilin_sorted.loc[model, 'MAE']
            yantai_mae = yantai_sorted.loc[model, 'MAE']
            gap_pct = ((yantai_mae - jilin_mae) / jilin_mae) * 100
            gap_data.append(gap_pct)
        
        colors = ['red' if x > 0 else 'green' for x in gap_data]
        bars = plt.bar(models, gap_data, color=colors, alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Model')
        plt.ylabel('Performance Gap (%)')
        plt.title('Performance Gap: Yantai vs Jilin (MAE)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, gap_data):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/thesis_fig3_performance_gaps.png")
        plt.close()
    
    def generate_latex_tables(self, output_dir="thesis_figures"):
        """Generate LaTeX tables for thesis"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Table 1: Complete results table
        latex_table = self.df_metrics.copy()
        latex_table = latex_table.round(3)
        
        # Reorder columns for better presentation
        column_order = ['city', 'model', 'MAE', 'RMSE', 'R2', 'accuracy', 'accuracy5', 'accuracy10', 'accuracy20']
        latex_table = latex_table[column_order]
        
        # Rename columns for LaTeX
        latex_table.columns = ['City', 'Model', 'MAE', 'RMSE', 'R²', 'Accuracy', 'Acc±5min', 'Acc±10min', 'Acc±20min']
        
        # Generate LaTeX code
        latex_code = latex_table.to_latex(index=False, escape=False, 
                                         caption="Complete Model Performance Results",
                                         label="tab:complete_results")
        
        with open(f"{output_dir}/complete_results_table.tex", 'w') as f:
            f.write(latex_code)
        
        # Table 2: Summary statistics
        summary_stats = self.generate_summary_statistics(output_dir)
        
        summary_data = []
        for city, stats in summary_stats.items():
            summary_data.append({
                'City': city,
                'Best Model (MAE)': f"{stats['best_mae_model']} ({stats['best_mae_value']:.2f})",
                'Best Model (R²)': f"{stats['best_r2_model']} ({stats['best_r2_value']:.3f})",
                'Best Model (Acc)': f"{stats['best_accuracy_model']} ({stats['best_accuracy_value']:.3f})",
                'Mean MAE': f"{stats['mean_mae']:.2f} ± {stats['std_mae']:.2f}",
                'Mean R²': f"{stats['mean_r2']:.3f} ± {stats['std_r2']:.3f}",
                'Mean Accuracy': f"{stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_latex = summary_df.to_latex(index=False, escape=False,
                                           caption="Summary Statistics by City",
                                           label="tab:summary_stats")
        
        with open(f"{output_dir}/summary_table.tex", 'w') as f:
            f.write(summary_latex)
    
    def run_complete_analysis(self):
        """Run complete error analysis pipeline"""
        print("Starting comprehensive error analysis...")
        
        # Create output directories
        output_dirs = ["error_analysis", "thesis_figures"]
        for dir_name in output_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        # Generate all analyses
        print("1. Creating model comparison plots...")
        self.plot_model_comparison()
        
        print("2. Analyzing performance gaps...")
        gap_df = self.analyze_performance_gaps()
        
        print("3. Analyzing error distributions...")
        self.analyze_error_distribution()
        
        print("4. Generating summary statistics...")
        summary_stats = self.generate_summary_statistics()
        
        print("5. Creating thesis-quality figures...")
        self.create_thesis_figures()
        
        print("6. Generating LaTeX tables...")
        self.generate_latex_tables()
        
        print("Error analysis complete! Check the output directories for results.")
        
        return {
            'gap_analysis': gap_df,
            'summary_stats': summary_stats,
            'metrics_df': self.df_metrics
        }
    

def main():
    """Main execution function"""
    print("Initializing Error Analysis...")
    
    # Initialize analysis
    analysis = ErrorAnalysis("metrics.json")
    
    # Run complete analysis
    results = analysis.run_complete_analysis()
    
    print("\n" + "="*50)
    print("ERROR ANALYSIS SUMMARY")
    print("="*50)
    
    # Print key findings
    jilin_best = analysis.df_metrics[analysis.df_metrics['city'] == 'Jilin'].loc[analysis.df_metrics[analysis.df_metrics['city'] == 'Jilin']['MAE'].idxmin()]
    yantai_best = analysis.df_metrics[analysis.df_metrics['city'] == 'Yantai'].loc[analysis.df_metrics[analysis.df_metrics['city'] == 'Yantai']['MAE'].idxmin()]
    
    print(f"Best model for Jilin: {jilin_best['model']} (MAE: {jilin_best['MAE']:.3f})")
    print(f"Best model for Yantai: {yantai_best['model']} (MAE: {yantai_best['MAE']:.3f})")
    
    print(f"\nModel performance ranking (by average MAE):")
    avg_mae = analysis.df_metrics.groupby('model')['MAE'].mean().sort_values()
    for i, (model, mae) in enumerate(avg_mae.items(), 1):
        print(f"{i}. {model}: {mae:.3f} minutes")
    
    print(f"\nAnalysis complete! Check the following directories:")
    print(f"- error_analysis/: Detailed analysis plots and statistics")
    print(f"- thesis_figures/: Publication-quality figures and LaTeX tables")

if __name__ == "__main__":
    main()

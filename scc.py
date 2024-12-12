import pandas as pd
import scienceplots
from matplotlib import pyplot as plt
import numpy as np
import argparse
from scipy.integrate import trapezoid
import yaml
plt.style.use(['science', 'nature', 'bright'])

class Settings:
    def __init__(self):
        self.data_directory = ''
        self.raw_data = []
        self.score_combination_weighted = []
        self.score_combination = []
        self.classify_molecules = False
        self.final_data = {}
        self.final_results = {}
        self.plot_classification = False
        self.save_name = ''
        self.save_dir = ''
        self.probability_data = []
        self.y_start = 0.7

class BaseAnalyser:
    """Base class for analysis containing common functionality"""
    def __init__(self, yaml_file_path='config.yaml'):
        self.settings = Settings()
        self.load_settings_from_yaml(yaml_file_path)
    
    def execute_strategy(self):
        """
        Executes analysis operations based on the configuration defined in settings.
        """
        operations = [
            {"condition": self.settings.raw_data, "method": self.process_raw_data},
            {"condition": self.settings.score_combination_weighted, "method": self.combine_scores_weighted},
            {"condition": self.settings.score_combination, "method": self.combine_scores},
            {"condition": self.settings.classify_molecules, "method": self.analyse_data},
            {"condition": self.settings.plot_classification, "method": self.plot_results},
        ]
        for op in operations:
            if op["condition"]:
                op["method"]()

    def load_settings_from_yaml(self, yaml_file_path):
        """Load settings from YAML configuration file."""
        try:
            with open(yaml_file_path, 'r') as file:
                config = yaml.safe_load(file)
                for key, value in config.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
        except Exception as e:
            print(f"Failed to load or parse YAML config file: {e}")

    def process_raw_data(self):
        """Process raw data files specified in settings."""
        for data in self.settings.raw_data:
            pairwise = self.convert_raw_data_pairwise(f'{self.settings.data_directory}/{data}.csv')
            probability = data in self.settings.probability_data
            pairwise = self.subtract_pairwise_scores(pairwise, probability=probability)
            self.settings.final_data[data] = pairwise

    @staticmethod
    def convert_raw_data_pairwise(data_path):
        df = pd.read_csv(data_path)

        # Splitting the 'Comparison' column to get individual molecule identifiers
        molecule_data = df['Comparison'].astype(str).str.split(':', expand=True)
        num_molecules = molecule_data.shape[1]  # Get the number of molecules dynamically
        molecule_columns = [f'Molecule_{i}' for i in range(num_molecules)]

        df = pd.concat([df, molecule_data], axis=1)
        df.columns = df.columns.tolist()[:-num_molecules] + molecule_columns

        # Creating a new df to store pairwise comparisons
        pairwise_comparisons = []

        # Looping over each row to create pairwise comparisons
        for index, row in df.iterrows():
            # Getting the base molecule ID (without _0, _1, etc.)
            base_molecule_id = row['Molecule_0'].split('_')[0]
            for j in range(1, num_molecules):  # Ensure we don't compare the molecule with itself
                if pd.notna(row[molecule_columns[j]]):
                    comparison_id = f"{base_molecule_id}_0:{base_molecule_id}_{j}"
                    score_0 = row['0']
                    score_decoy = row[str(j)]
                    pairwise_comparisons.append([comparison_id,score_0,score_decoy])
        pairwise_df = pd.DataFrame(pairwise_comparisons, columns=['Comparison', '0', '1'])

        return pairwise_df
    
    @staticmethod
    def subtract_pairwise_scores(df, probability=False):
        if probability:
            df['score'] = (df['0'] - df['1']) / (df['0'] + df['1'])
        else:
            df['score'] = (df['0'] - df['1'])
        
        return df[['Comparison', 'score']]

    def combine_scores_weighted(self):
        """Combine scores with given weighting."""
        for method1, method2, weighting in self.settings.score_combination_weighted:
            combined_method_name = f'{float(weighting):.2f}{method1}+{(1 - float(weighting)):.2f}{method2}'
            combined_scores = self.combine_scores_weighted_processing(self.settings.final_data[method1], self.settings.final_data[method2], weighting)
            self.settings.final_data[combined_method_name] = combined_scores
    
    def combine_scores(self):
        """Combine scores with equal weighting"""
        for method_group in self.settings.score_combination:
            combined_method_name = '+'.join(method_group)
            combined_scores = self.combine_scores_processing(*[self.settings.final_data[method] for method in method_group])
            self.settings.final_data[combined_method_name] = combined_scores

    @staticmethod
    def combine_scores_weighted_processing(df1, df2, weight_1):
        merged_df = df1
        merged_df = pd.merge(merged_df, df2, on='Comparison', suffixes=('', '1'))
        weight_1 = float(weight_1)
        merged_df['combined_score'] = weight_1 * merged_df['score'] + (1 - weight_1) * merged_df['score1']
        merged_df = merged_df[['Comparison', 'combined_score']].rename(columns={'combined_score': 'score'})
        return merged_df
    
    @staticmethod
    def combine_scores_processing(*dataframes):
        merged_df = dataframes[0]

        for idx, df in enumerate(dataframes[1:]):
            merged_df = pd.merge(merged_df, df, on='Comparison', suffixes=('', str(idx + 1)))

        score_columns = [col for col in merged_df.columns if col != 'Comparison']
        merged_df['combined_score'] = merged_df[score_columns].mean(axis=1)
        merged_df = merged_df[['Comparison', 'combined_score']].rename(columns={'combined_score': 'score'})
        return merged_df

    def analyse_data(self):
        """To be implemented by child classes"""
        raise NotImplementedError

    def plot_results(self):
        """To be implemented by child classes"""
        raise NotImplementedError

    def run_analysis(self):
        self.execute_strategy()


class SCC_Analyser(BaseAnalyser):
    def analyse_data(self):
        """Classify molecules and find classification rates."""
        if self.settings.classify_molecules:
            for method, data in self.settings.final_data.items():
                results_ryg = self.classify_molecules(data)
                classification_results = self.find_classification_rates(results_ryg)
                self.settings.final_data[method] = classification_results

    @staticmethod
    def classify_molecules(df):
        # Define the thresholds
        normal_thresholds = np.arange(0, 1, 0.0005)
        additional_thresholds = np.array([0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999])
        thresholds = np.concatenate((normal_thresholds, additional_thresholds))
        thresholds = np.unique(np.sort(thresholds))

        def count_scores(series, threshold):
            green = series[(series > 0) & (series > threshold)].count()
            yellow = series[series.abs() < threshold].count()
            red = series[(series < 0) & (series.abs() > threshold)].count()

            guess = False
            if threshold == 0: 
                equal_threshold_count = series[series.abs() == threshold].count()
                green_add = equal_threshold_count // 2
                red_add = equal_threshold_count - green_add
                green += green_add
                red += red_add
                if red_add:
                    guess = True

            return green, yellow, red, guess

        results = pd.DataFrame([count_scores(df['score'], t) for t in thresholds], columns=["Green", "Yellow", "Red", "Guess"])
        results.index = thresholds

        return results

    @staticmethod
    def find_classification_rates(classifier_scores):
        classifier_scores['Classified'] = (classifier_scores['Green'] + classifier_scores['Red']) / (classifier_scores['Green'] + classifier_scores['Yellow'] + classifier_scores['Red']) 
        classifier_scores['Correct'] = classifier_scores['Green'] / (classifier_scores['Green'] + classifier_scores['Red'])

        classification_results = classifier_scores[['Classified', 'Correct', 'Guess']].copy()

        zero_classified_row = pd.DataFrame({
            'Classified': [0],
            'Correct': [1],
            'Guess': [0]
        })
        classification_results = pd.concat([classification_results, zero_classified_row], ignore_index=True)

        return classification_results

    def plot_results(self):
        """Plot all classifications if enabled."""
        if self.settings.plot_classification:
            self.plot_classification(self.settings)

    @staticmethod
    def plot_classification(settings):
        plt.figure(figsize=(6,3))
        colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for idx, (method, data) in enumerate(settings.final_data.items()):
            data = data.dropna(subset=['Correct', 'Classified']).copy()
            data.sort_values('Classified', inplace=True)
            
            auc = trapezoid(y=data['Correct'], x=data['Classified'])
            aoc = 1 - auc

            current_colour = colour_cycle[idx % len(colour_cycle)]

            data_false = data[data['Guess'] == False]
            plt.plot(data_false['Classified'], data_false['Correct'], color=current_colour, linewidth=2, label=f"{method}; CA = {round(auc, 3)}")
            
            prev_row = None
            for idx, row in data.iterrows():
                if row['Guess']:
                    if prev_row is not None:
                        plt.plot([prev_row['Classified'], row['Classified']],
                                [prev_row['Correct'], row['Correct']],
                                linestyle=':', color=current_colour, linewidth=2, zorder=10)
                prev_row = row

        plt.xlabel('Classification rate')
        plt.ylabel('Fraction of classifications correct')
        plt.title('Fraction of classifications correct vs classification rate for various conditions')
        plt.xlim(0, 1)
        plt.ylim(settings.y_start,1.01)
        plt.legend(loc='lower left')
        plt.tight_layout()
        if settings.save_name and settings.save_dir:
            save_path = f'{settings.save_dir}/{settings.save_name}.pdf'
            plt.savefig(save_path)
        else:
            plt.show()


class ROC_Analyser(BaseAnalyser):
    @staticmethod
    def subtract_pairwise_scores(df, probability=False):
        """Override to keep all scores separately"""
        # Keep all score columns and the comparison identifier
        score_columns = [col for col in df.columns if col.isdigit()]
        return df[['Comparison'] + score_columns]
    
    def analyse_data(self):
        """Calculate ROC curve points for each method."""
        if self.settings.classify_molecules:
            for method, data in self.settings.final_data.items():
                roc_points = self.calculate_single_roc(data)
                self.settings.final_data[method] = roc_points

    @staticmethod
    def calculate_single_roc(df):
        """Calculate TPR and FPR for different thresholds."""
        # Get all score columns (those that are numeric)
        score_columns = [col for col in df.columns if col.isdigit()]
        decoy_columns = score_columns[1:]  # All columns except '0'
        
        # Generate thresholds from all scores
        all_scores = pd.concat([df[col] for col in score_columns])
        thresholds = np.concatenate([
            np.arange(all_scores.max(), all_scores.min(), -0.001),
            [all_scores.min()]
        ])
        
        results = []
        total_positives = len(df)  # One correct structure per comparison
        total_negatives = df[decoy_columns].count().sum()
        
        for threshold in thresholds:
            # True Positives: correct structures (col '0') above threshold
            tp = (df['0'] > threshold).sum()
            
            # False Positives: any decoy structure above threshold
            fp = sum((df[col] > threshold).sum() for col in decoy_columns)
            
            tpr = tp / total_positives if total_positives > 0 else 0
            fpr = fp / total_negatives if total_negatives > 0 else 0
            
            results.append([fpr, tpr])
        
        return pd.DataFrame(results, columns=['FPR', 'TPR'])

    def plot_results(self):
        """Plot ROC curves if enabled."""
        if self.settings.plot_classification:
            self.plot_roc_curves(self.settings)

    @staticmethod
    def plot_roc_curves(settings):
        plt.figure(figsize=(6, 3))
        colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        
        for idx, (method, data) in enumerate(settings.final_data.items()):
            auc = trapezoid(y=data['TPR'], x=data['FPR'])
            current_colour = colour_cycle[idx % len(colour_cycle)]
            plt.plot(data['FPR'], data['TPR'], 
                    color=current_colour, 
                    linewidth=2, 
                    label=f"{method}; AUC = {round(auc, 3)}")

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves for various conditions')
        plt.xlim(0, 1)
        plt.ylim(0, 1.01)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        if settings.save_name and settings.save_dir:
            save_path = f'{settings.save_dir}/{settings.save_name}_roc.pdf'
            plt.savefig(save_path)
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analysis")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--type', type=str, choices=['scc', 'roc'], default='scc', help='Type of analysis to perform')
    args = parser.parse_args()
    
    analyser_class = SCC_Analyser if args.type == 'scc' else ROC_Analyser
    if args.config:
        analyser = analyser_class(args.config)
    else:
        analyser = analyser_class()
    analyser.run_analysis()

if __name__ == '__main__':
    main()
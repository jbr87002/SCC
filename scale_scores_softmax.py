import pandas as pd
import numpy as np
import argparse

def softmax(scores, T=1.0):
    scores = np.array(scores)
    max_score = np.nanmax(scores, axis=1, keepdims=True)
    scores = scores - max_score
    exp_scores = np.exp(scores / T)
    sum_exp_scores = np.nansum(exp_scores, axis=1, keepdims=True)
    softmax_scores = exp_scores / sum_exp_scores
    return np.where(np.isnan(scores), np.nan, softmax_scores)

def calculate_average_sd(data):
    return np.nanmean(np.nanstd(data, axis=1))

def adjust_probabilities_to_match_sd(source_data, target_sd, T_initial=1.0, tolerance=0.001, DP4=True):
    """Adjust source data to match target standard deviation using softmax with temperature scaling."""
    T = T_initial
    # obtain 'latent logits' by taking log if working with DP4 scores
    if DP4:
        estimated_scores = np.log(source_data + 1e-20)
    else:
        estimated_scores = source_data
    
    current_sd = calculate_average_sd(estimated_scores)
    while np.abs(target_sd - current_sd) > tolerance:
        adjusted_data = softmax(estimated_scores, T=T)
        current_sd = calculate_average_sd(adjusted_data)
        if current_sd < target_sd:
            T /= 1.05  # Decrease temperature to increase spread
        else:
            T *= 1.05  # Increase temperature to decrease spread

    return adjusted_data, T

def main(args):
    # apply softmax to target and save to target_output
    score_columns = ['0', '1', '2', '3']
    df1 = pd.read_csv(args.target) 
    data1 = df1[score_columns].values
    if 'DP4' in args.target.split('/')[-1]:
        print('No need to softmax DP4* scores, they are already a probability distribution')
        softmaxed_data1 = data1
    else:
        softmaxed_data1 = softmax(data1)
        df1[score_columns] = softmaxed_data1
        df1.to_csv(args.target_output, index=False)
        print(f"Softmaxed target saved to {args.target_output}")
    
    # read in the data to scale to target
    df2 = pd.read_csv(args.input_file)
    data2 = df2[score_columns].values

    # Calculate the average standard deviation of the softmaxed target
    avg_sd_data1 = calculate_average_sd(softmaxed_data1)
    DP4 = 'DP4' in args.input_file

    # Adjust the scores in the second method to match the average st dev of the first
    adjusted_data2, T_optimal = adjust_probabilities_to_match_sd(data2, avg_sd_data1, DP4=DP4)

    df2[score_columns] = adjusted_data2
    df2.to_csv(args.output_file, index=False)

    print(f"Required temperature: {T_optimal}")
    print(f"Scaled output saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust scores to match the standard deviation of a target dataset.")
    parser.add_argument('--target', type=str, required=True, help="Path to target csv (to match st dev of)")
    parser.add_argument('--target_output', type=str, required=True, help="Path to save softmaxed target data")
    parser.add_argument('--input_file', type=str, required=True, help="Path to input csv")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save softmax scaled data")

    args = parser.parse_args()
    main(args)

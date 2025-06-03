import argparse
import pandas as pd
from tqdm import tqdm
import os
import sys
from pathlib import Path
import math
from shared.src.models import ModelLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run log probability inference on sentence stimuli using minicons.")
    parser.add_argument("--stimuli_csv", type=str, required=True, help="Path to CSV file with stimuli.")
    parser.add_argument("--model_identifiers", nargs="+", required=True, 
                        help="List of Hugging Face model IDs or local paths to models (e.g., 'gpt2' '/path/to/my-model').")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token (if required by models). Can also be set via HF_TOKEN env var.")
    parser.add_argument("--quantization_bits", type=int, default=None, choices=[4, 8], help="Quantization bits (4 or 8). Default: None.")
    parser.add_argument("--fast_inference", action="store_true", help="Enable fast inference optimizations (e.g., specific dtypes if not quantizing).")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model loading (e.g., 'cuda', 'cpu', 'auto').")
    parser.add_argument("--scorer_device", type=str, default="cuda", help="Device for minicons scorer operations (e.g., 'cuda', 'cpu').")
    parser.add_argument("--mode", type=str, default="sentence", choices=["sentence", "conditional"], help="Scoring mode: 'sentence' for P(sentence), 'conditional' for P(interpretation|preamble).")
    parser.add_argument("--sentence_column", type=str, default="sentence", help="Column name in CSV for sentences (used in 'sentence' mode).")
    parser.add_argument("--preamble_column", type=str, default="preamble", help="Column for preamble (context) in 'conditional' mode.")
    parser.add_argument("--interpretation_A_column", type=str, default="interpretation_A", help="Column for the first interpretation in 'conditional' mode.")
    parser.add_argument("--interpretation_B_column", type=str, default="interpretation_B", help="Column for the second interpretation in 'conditional' mode.")
    parser.add_argument("--apply_sigmoid_to_diff", action="store_true", help="In 'conditional' mode, apply sigmoid to logprob_B - logprob_A.")


    return parser.parse_args()

def calculate_sentence_logprobs(scorer, sentences: list) -> list:
    """Calculates sum of log_e probabilities for a list of sentences."""
    logger.info(f"Calculating log probabilities for {len(sentences)} sentences...")
    log_probs = scorer.score_batch(sentences, add_bos_token=True)  # add_bos_token is often good.
    return log_probs

def calculate_conditional_logprobs(scorer, preambles: list, interpretations_A: list, interpretations_B: list) -> tuple[list, list]:
    """
    Calculates log P(interpretation_A | preamble) and log P(interpretation_B | preamble).
    Uses minicons' conditional_score, which expects one preamble and a list of candidates.
    """
    logger.info(f"Calculating conditional log probabilities for {len(preambles)} sets of stimuli...")
    scores_A = []
    scores_B = []

    for i in tqdm(range(len(preambles)), desc="Processing conditional stimuli"):
        preamble = str(preambles[i]) # Ensure string type
        interp_A = str(interpretations_A[i])
        interp_B = str(interpretations_B[i])
        
        # conditional_score returns: [log P(interp_A | preamble), log P(interp_B | preamble)]
        try:
            current_scores = scorer.conditional_score(preamble, [interp_A, interp_B])
            scores_A.append(current_scores[0])
            scores_B.append(current_scores[1])
        except Exception as e:
            logger.error(f"Error scoring preamble '{preamble[:50]}...' with interpretations '{interp_A[:50]}...', '{interp_B[:50]}...': {e}")
            scores_A.append(None) # Append None or NaN for error cases
            scores_B.append(None)
            
    return scores_A, scores_B

def sigmoid(z: float) -> float:
    if z is None: return None
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError: # Handle very large or very small z
        return 0.0 if z < 0 else 1.0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        stimuli_df = pd.read_csv(args.stimuli_csv)
        logger.info(f"Successfully loaded stimuli from {args.stimuli_csv} with {len(stimuli_df)} rows.")
    except FileNotFoundError:
        logger.error(f"Stimuli CSV file not found: {args.stimuli_csv}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading stimuli CSV '{args.stimuli_csv}': {e}")
        sys.exit(1)

    # Validate columns based on mode
    if args.mode == "sentence":
        if args.sentence_column not in stimuli_df.columns:
            logger.error(f"Sentence column '{args.sentence_column}' not found in {args.stimuli_csv} for 'sentence' mode.")
            sys.exit(1)
    elif args.mode == "conditional":
        required_cols = [args.preamble_column, args.interpretation_A_column, args.interpretation_B_column]
        for col in required_cols:
            if col not in stimuli_df.columns:
                logger.error(f"Required column '{col}' not found in {args.stimuli_csv} for 'conditional' mode.")
                sys.exit(1)

    model_loader = ModelLoader(hf_token=args.hf_token)
    
    all_model_results_dfs = []

    for model_id_or_path in args.model_identifiers:
        logger.info(f"--- Processing model: {model_id_or_path} ---")
        try:
            model_loader.load_model_and_tokenizer(
                model_id_or_path,
                fast_inference=args.fast_inference,
                quantization_bits=args.quantization_bits,
                device=args.device,
                trust_remote_code=args.trust_remote_code
            )
            scorer = model_loader.get_scorer(scorer_device=args.scorer_device)
        except Exception as e:
            logger.error(f"Failed to load model {model_id_or_path} or get scorer: {e}. Skipping this model.")
            error_df = stimuli_df.copy()
            error_df['model_key'] = model_id_or_path
            error_df['error'] = str(e)
            all_model_results_dfs.append(error_df)
            continue

        current_model_results_df = stimuli_df.copy()
        current_model_results_df['model_key'] = model_id_or_path
        
        if args.mode == "sentence":
            sentences = current_model_results_df[args.sentence_column].astype(str).tolist()
            sentence_log_probs = calculate_sentence_logprobs(scorer, sentences)
            current_model_results_df['sentence_log_probability'] = sentence_log_probs
        
        elif args.mode == "conditional":
            preambles = current_model_results_df[args.preamble_column].astype(str).tolist()
            interpretations_A = current_model_results_df[args.interpretation_A_column].astype(str).tolist()
            interpretations_B = current_model_results_df[args.interpretation_B_column].astype(str).tolist()
            
            logprobs_A, logprobs_B = calculate_conditional_logprobs(scorer, preambles, interpretations_A, interpretations_B)
            
            current_model_results_df['logprob_A_given_preamble'] = logprobs_A
            current_model_results_df['logprob_B_given_preamble'] = logprobs_B
            
            # Calculate difference (B vs A, similar to original script's ratio if A is surface, B is inverse)
            diff_B_vs_A = [(b - a) if a is not None and b is not None else None for a, b in zip(logprobs_A, logprobs_B)]
            current_model_results_df['logprob_diff_B_vs_A'] = diff_B_vs_A

            if args.apply_sigmoid_to_diff:
                current_model_results_df['sigmoid_logprob_diff_B_vs_A'] = [sigmoid(d) for d in diff_B_vs_A]

        all_model_results_dfs.append(current_model_results_df)
        logger.info(f"Finished processing for model {model_id_or_path}.")
    
    if all_model_results_dfs:
        final_results_df = pd.concat(all_model_results_dfs, ignore_index=True)
        output_filename = "logprob_inference_results.csv"
        output_path = os.path.join(args.output_dir, output_filename)
        final_results_df.to_csv(output_path, index=False)
        logger.info(f"All results saved to {output_path}")
    else:
        logger.warning("No models were processed successfully. No results to save.")
    
    if model_loader._current_model_key:
        model_loader.unload_model()

    logger.info("--- Log probability inference script finished. ---")

if __name__ == "__main__":
    main()
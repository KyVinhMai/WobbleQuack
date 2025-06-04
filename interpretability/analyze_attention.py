import argparse
import torch
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import logging
import math
from shared.src.models import ModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_requested_indices(requested_spec: list, total_available: int, item_type: str) -> list:
    """
    Parses user-specified layer or head indices.
    Args:
        requested_spec (list): List of strings from argparse (e.g., ["all"], ["0", "5"]).
        total_available (int): Total number of available items (layers or heads).
        item_type (str): Type of item being parsed ("layer" or "head") for logging.
    Returns:
        list: Sorted list of integer indices to process.
    """
    if not requested_spec or not total_available:
        logger.warning(f"No {item_type}s available or specified. Returning empty list.")
        return []

    if "all" in [str(s).lower() for s in requested_spec]:
        return list(range(total_available))
    
    indices = set()
    for spec_item in requested_spec:
        try:
            idx = int(spec_item)
            if 0 <= idx < total_available:
                indices.add(idx)
            else:
                logger.warning(f"Invalid {item_type} index {idx} (out of range 0-{total_available-1}). Skipping.")
        except ValueError:
            logger.warning(f"Invalid {item_type} specifier '{spec_item}'. Expected 'all' or integer. Skipping.")
    
    if not indices:
        # This case might occur if all specific inputs were invalid, and "all" was not specified.
        # Depending on desired behavior, could default to all, or error, or return empty.
        # For now, let's warn and return empty if no valid specific indices were found.
        logger.warning(f"No valid specific {item_type} indices found in '{requested_spec}'. Check your input.")
        return []
        
    return sorted(list(indices))


def plot_attention_heatmap(attention_matrix, tokens, model_name_display, layer_idx, head_idx, 
                           input_text_snippet, output_dir, figsize, dpi):
    """
    Plots and saves an attention heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention_matrix, cmap="viridis")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")

    title_text = f"Model: {model_name_display}, Layer: {layer_idx}, Head: {head_idx}"
    if len(input_text_snippet) < 80:
        title_text += f"\nInput: '{input_text_snippet}'"
    else:
        title_text += f"\nInput (start): '{input_text_snippet[:77]}...'"

    ax.set_title(title_text, fontsize=10)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filename_base = f"attention_l{layer_idx}_h{head_idx}"
    filename = Path(output_dir) / f"{filename_base}.png"
    
    try:
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        logger.info(f"Saved attention heatmap to {filename}")
    except Exception as e:
        logger.error(f"Failed to save plot {filename}: {e}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and visualize attention patterns from transformer models.")
    parser.add_argument("--model_identifier", type=str, required=True,
                        help="Hugging Face model ID or local path to the model.")
    parser.add_argument("--input_text", type=str, required=True,
                        help="Text stimulus to analyze.")
    parser.add_argument("--layers", nargs="+", type=str, default=["all"],
                        help="Layers to analyze (e.g., 'all', '0', '5', '10'). Default: 'all'.")
    parser.add_argument("--heads", nargs="+", type=str, default=["all"],
                        help="Attention heads to analyze for each specified layer (e.g., 'all', '0', '1'). Default: 'all'.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save attention visualizations.")
    
    # ModelLoader arguments
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face API token (if required).")
    parser.add_argument("--quantization_bits", type=int, default=None, choices=[4, 8],
                        help="Quantization bits (4 or 8). Default: None.")
    parser.add_argument("--fast_inference", action="store_true",
                        help="Enable fast inference optimizations (e.g., specific dtypes).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for model loading ('cuda', 'cpu', 'auto'). Default: 'cuda' if available, else 'cpu'.")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Set to True if loading a model that requires trusting remote code.")

    # Plotting arguments
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 8],
                        help="Figure size (width, height) in inches for plots. Default: 10 8.")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved figures. Default: 150.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    model_name_display = Path(args.model_identifier).name

    logger.info(f"Starting attention analysis for model: {args.model_identifier}")
    logger.info(f"Input text: '{args.input_text}'")
    logger.info(f"Output directory: {args.output_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model_loader = ModelLoader(hf_token=args.hf_token)
    model = None

    try:
        model, tokenizer = model_loader.load_model_and_tokenizer(
            args.model_identifier,
            fast_inference=args.fast_inference,
            quantization_bits=args.quantization_bits,
            device=args.device,
            trust_remote_code=args.trust_remote_code
        )
        model.eval()

        # Tokenize input
        inputs = tokenizer(args.input_text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0]) # Get token strings for labels

        if not tokens:
            logger.error("Tokenization resulted in no tokens. Cannot proceed.")
            return

        # Perform forward pass to get attentions
        with torch.no_grad():
            model_outputs = model(input_ids, output_attentions=True)
        
        attentions = model_outputs.attentions # Tuple of tensors, one for each layer

        if attentions is None:
            logger.error("Model did not output attentions. Ensure the model architecture supports it and 'output_attentions=True' is effective.")
            return

        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads

        layers_to_analyze = parse_requested_indices(args.layers, num_layers, "layer")
        heads_to_analyze_for_each_layer = parse_requested_indices(args.heads, num_heads, "head")

        if not layers_to_analyze:
            logger.warning("No layers selected for analysis. Exiting.")
            return
        if not heads_to_analyze_for_each_layer:
            logger.warning("No heads selected for analysis. Exiting.")
            return

        logger.info(f"Analyzing layers: {layers_to_analyze}")
        logger.info(f"Analyzing heads (for each selected layer): {heads_to_analyze_for_each_layer}")

        for layer_idx in layers_to_analyze:
            if layer_idx < 0 or layer_idx >= len(attentions):
                logger.warning(f"Layer index {layer_idx} is out of bounds for attentions tuple. Skipping.")
                continue
            
            # attentions[layer_idx] shape: (batch_size, num_heads, seq_len, seq_len)
            # Squeeze batch_size (assuming it's 1)
            layer_attention_tensor = attentions[layer_idx].squeeze(0).cpu() # Move to CPU for numpy conversion

            for head_idx in heads_to_analyze_for_each_layer:
                if head_idx < 0 or head_idx >= layer_attention_tensor.shape[0]: # num_heads is shape[0] after squeeze
                    logger.warning(f"Head index {head_idx} is out of bounds for layer {layer_idx}. Skipping.")
                    continue

                head_attention_matrix = layer_attention_tensor[head_idx].numpy() # (seq_len, seq_len)

                plot_attention_heatmap(
                    head_attention_matrix, tokens, model_name_display, layer_idx, head_idx,
                    args.input_text, args.output_dir, tuple(args.figsize), args.dpi
                )
        
        logger.info("Attention analysis script finished.")

    except Exception as e:
        logger.error(f"An error occurred during attention analysis: {e}", exc_info=True)
    finally:
        if hasattr(model_loader, '_current_model_identifier') and model_loader._current_model_identifier:
            logger.info(f"Attempting to unload model: {model_loader._current_model_identifier}")
            model_loader.unload_model()

if __name__ == "__main__":
    main()
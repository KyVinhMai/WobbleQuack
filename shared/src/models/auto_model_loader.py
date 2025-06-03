import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from minicons import scorer as minicons_scorer
import logging
import gc
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, hf_token: str = None):
        """
        Initializes the ModelLoader.
        Args:
            hf_token (str, optional): Hugging Face API token. Used for downloading models/tokenizers
                                      from the Hugging Face Hub, especially gated ones.
                                      Can also be set via HF_TOKEN environment variable.
        """
        self.current_model = None
        self.tokenizer = None
        self.scorer = None
        self._current_model_identifier = None # Stores the HF ID or path of the loaded model
        
        # Prioritize explicitly passed token, then environment variable
        if hf_token:
            self.hf_token = hf_token
            logger.info("Using Hugging Face token provided via argument.")
        elif os.getenv("HF_TOKEN"):
            self.hf_token = os.getenv("HF_TOKEN")
            logger.info("Using Hugging Face token from HF_TOKEN environment variable.")
        else:
            self.hf_token = None
            logger.info("No Hugging Face token provided or found in HF_TOKEN environment variable. Public models only unless cached.")
        logger.info("ModelLoader initialized.")

    def _check_resources(self, model_identifier: str):
        """
        Basic placeholder for checking if system resources are adequate for the model.
        Args:
            model_identifier (str): The Hugging Face ID or local path of the model.
        """
        # TODO: Implement more sophisticated resource checking if needed.
        # This is difficult without model-specific metadata for arbitrary paths/IDs.
        logger.info(f"Resource check for model '{model_identifier}' - (Placeholder: No specific checks implemented).")
        pass

    def unload_model(self):
        """
        Unloads the currently loaded model and tokenizer, and clears CUDA cache.
        * Might be used in experiments where you want the model to have no memory of previous experiment/conversation
        """
        if self._current_model_identifier:
            logger.info(f"Unloading model: {self._current_model_identifier}...")
        else:
            logger.info("No model currently loaded to unload.")
            return

        if self.scorer:
            del self.scorer
            self.scorer = None
        if self.current_model:
            del self.current_model
            self.current_model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        unloaded_model_identifier = self._current_model_identifier
        self._current_model_identifier = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared after unloading.")
        gc.collect()
        logger.info(f"Model {unloaded_model_identifier} unloaded successfully.")

    def load_model_and_tokenizer(self, model_identifier: str, fast_inference: bool = False, 
                                 quantization_bits: int = None, device: str = "cuda",
                                 trust_remote_code: bool = False): # Added trust_remote_code
        """
        Load model and tokenizer from a Hugging Face model ID or a local path.
        Args:
            model_identifier (str): Hugging Face model ID (e.g., "meta-llama/Meta-Llama-3-8B") 
                                     or path to a local directory containing model files.
            fast_inference (bool): General flag for optimizations. Influences torch_dtype if no specific quantization.
            quantization_bits (int, optional): Load in specified bits (e.g., 4 or 8). None for no quantization.
            device (str): "cuda", "cpu", or "auto" for device_map.
            trust_remote_code (bool): Whether to trust remote code for models that require it. Defaults to False.
        Returns:
            tuple: (model, tokenizer)
        """
        if not model_identifier:
            raise ValueError("model_identifier cannot be empty.")

        if self._current_model_identifier == model_identifier and self.current_model and self.tokenizer:
            logger.info(f"Model '{model_identifier}' is already loaded.")
            return self.current_model, self.tokenizer

        if self._current_model_identifier is not None:
            self.unload_model() # Unload previous model

        self._check_resources(model_identifier)
        
        is_local_path = os.path.exists(model_identifier)
        if is_local_path:
            logger.info(f"Loading model from local path: {model_identifier}...")
        else:
            logger.info(f"Loading model from Hugging Face Hub: {model_identifier}...")


        bnb_config = None
        model_kwargs = {
            "token": self.hf_token, # Pass token for HF Hub access; ignored if not needed or local
            "device_map": device if device != "auto" else "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": trust_remote_code,
        }
        tokenizer_kwargs = {
            "token": self.hf_token,
            "trust_remote_code": trust_remote_code,
        }


        if quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            logger.info("Applying 4-bit quantization.")
            # For 4-bit, device_map must be "auto" or a single GPU device.
            # If `device` is "cpu", quantization might not be supported or behave as expected.
            if device == "cpu" and bnb_config:
                logger.warning("4-bit quantization is typically used with CUDA. Behavior on CPU might be unexpected or unsupported.")
        elif quantization_bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Applying 8-bit quantization.")
            if device == "cpu" and bnb_config:
                logger.warning("8-bit quantization is typically used with CUDA. Behavior on CPU might be unexpected or unsupported.")


        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        elif fast_inference and torch.cuda.is_available() and not bnb_config:
             model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # Only apply torch_dtype if not quantizing, as quantization_config handles its own dtypes
             logger.info(f"Using torch_dtype: {model_kwargs['torch_dtype']} for fast inference.")


        try:
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                **model_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_identifier,
                use_fast=True,
                padding_side="left", # Consistent padding side for Causal LMs
                **tokenizer_kwargs
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.tokenizer.pad_token is None: # If eos_token is also None
                     logger.warning(f"EOS token is not set for tokenizer '{model_identifier}'. Using a default pad token ID (e.g., 0), but this may lead to issues. Please ensure your tokenizer has an EOS token or pad token defined.")
                     # Fallback, though this is risky. Models usually have EOS.
                     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                     if self.current_model.get_input_embeddings().weight.shape[0] < len(self.tokenizer):
                         self.current_model.resize_token_embeddings(len(self.tokenizer))

                logger.info("Tokenizer pad_token set to eos_token (or a fallback if eos was also None).")
            
            self._current_model_identifier = model_identifier
            logger.info(f"Successfully loaded model and tokenizer from '{model_identifier}'.")

        except Exception as e:
            self.unload_model()
            logger.error(f"Error loading model/tokenizer from '{model_identifier}': {e}")
            raise

        return self.current_model, self.tokenizer

    def get_scorer(self, scorer_device: str = "cuda"):
        """
        Returns a minicons IncrementalLMScorer for the currently loaded model.
        Args:
            scorer_device (str): The device on which the scorer should perform its computations.
                                 Minicons will handle data movement to this device.
        Returns:
            minicons.scorer.IncrementalLMScorer: The scorer instance.
        """
        if not self.current_model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before getting a scorer.")
        
        if self.scorer is None or self.scorer.model != self.current_model:
            logger.info(f"Creating/Recreating minicons scorer for {self._current_model_identifier} on device {scorer_device}.")
            
            # If model is on CPU and scorer_device is CUDA, warn or adjust.
            model_device_type = self.current_model.device.type
            if model_device_type == 'cpu' and scorer_device == 'cuda':
                logger.warning(f"Model '{self._current_model_identifier}' is loaded on CPU, but scorer_device is set to 'cuda'. Adjusting scorer_device to 'cpu'.")
                effective_scorer_device = 'cpu'
            else:
                effective_scorer_device = scorer_device

            self.scorer = minicons_scorer.IncrementalLMScorer(
                self.current_model,
                tokenizer=self.tokenizer,
                device=effective_scorer_device 
            )
        return self.scorer
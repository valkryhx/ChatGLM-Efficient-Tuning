import os
import torch
from typing import Dict, Optional

from transformers import Seq2SeqTrainer,Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from peft import PeftModel

from glmtuner.extras.constants import FINETUNING_ARGS_NAME, VALUE_HEAD_FILE_NAME
from glmtuner.extras.logging import get_logger
from glmtuner.extras.save_and_load import get_state_dict, load_trainable_params, load_valuehead_params
from glmtuner.hparams import FinetuningArguments
import torch.nn as nn

logger = get_logger(__name__)

"""https://github.com/thomasjpfan/pytorch/blob/e47af44eb81b9cd0c3583de91b0a2d4f56a5cf8d/torch/testing/_internal/common_fsdp.py#L111
"""
def _zero_model(
    model: nn.Module,
    zero_buffers: bool = False,
):
    """Zeros the parameters and optionally buffers of ``model`` in place."""
    for param in model.parameters():
        with torch.no_grad():
            param.zero_()
    if zero_buffers:
        for buffer in model.buffers():
            with torch.no_grad():
                buffer.zero_()


class PeftTrainer(Trainer):
    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        print("000X")
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        #self._remove_log()
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        print("111X")
        """只保存adapter"""
        if output_dir is None:
            output_dir = self.args.output_dir
        print("222X")
        """
        [bug discussion]Size of saved model checkpoints after trainer.train() is much larger when using trainer with deepspeed stage2
        https://github.com/thomasjpfan/pytorch/blob/e47af44eb81b9cd0c3583de91b0a2d4f56a5cf8d/torch/testing/_internal/common_fsdp.py#L872C1-L872C1
        https://github.com/microsoft/DeepSpeed/issues/3303
        https://github.com/huggingface/transformers/issues/22822#issuecomment-1514096667
        https://github.com/huggingface/transformers/issues/22822
        """
        state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        # Zero params, if save/load state_dict did not work properly, this
        # would break the parity test with DDP.
        _zero_model(self.model)
        print("333X")
        self.model.load_state_dict(state_dict)
        print("444X")
        self.model.save_pretrained(output_dir)
        print("555X")
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        print("666X")

class PeftTrainer2(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to support parameter-efficient checkpoints.
    """

    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self._remove_log()
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        print("99999999999999999999999999999999999999999999999999999999999999999")
        """只保存adapter"""
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
    def _remove_log(self):
        if self.is_world_process_zero() and os.path.exists(os.path.join(self.args.output_dir, "trainer_log.jsonl")):
            logger.warning("Previous log file in this folder will be deleted.")
            os.remove(os.path.join(self.args.output_dir, "trainer_log.jsonl"))

    def _save2(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoint.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = unwrap_model(self.model)

        if hasattr(model, "pretrained_model"): # for models with valuehead (currently using LoRA only)
            backbone_model = getattr(model, "pretrained_model")
            torch.save(get_state_dict(getattr(model, "v_head")), os.path.join(output_dir, VALUE_HEAD_FILE_NAME))
        else:
            backbone_model = model

        if isinstance(backbone_model, PeftModel): # LoRA tuning
            backbone_model.save_pretrained(output_dir, state_dict=get_state_dict(backbone_model))
        elif isinstance(backbone_model, PreTrainedModel): # freeze/full-tuning or p_tuning
            backbone_model.config.use_cache = True
            backbone_model.save_pretrained(
                output_dir,
                state_dict=get_state_dict(backbone_model, trainable_only=(self.finetuning_args.finetuning_type != "full")),
                safe_serialization=self.args.save_safetensors
            )
            backbone_model.config.use_cache = False
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
        else:
            logger.warning("No model to save.")

        with open(os.path.join(output_dir, TRAINING_ARGS_NAME), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string() + "\n")
        self.finetuning_args.save_to_json(os.path.join(output_dir, FINETUNING_ARGS_NAME))

    def _load_best_model(self):
        r"""
        Loads trainable parameters from model checkpoint.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")

        model = unwrap_model(self.model)
        backbone_model = getattr(model, "pretrained_model") if hasattr(model, "pretrained_model") else model

        if isinstance(backbone_model, PeftModel):
            backbone_model.load_adapter(self.state.best_model_checkpoint, backbone_model.active_adapter)
            if hasattr(model, "v_head") and load_valuehead_params(model, self.state.best_model_checkpoint):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })
        else: # freeze/full-tuning or p_tuning
            load_trainable_params(backbone_model, self.state.best_model_checkpoint)

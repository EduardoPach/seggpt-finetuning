from typing import Optional

import torch
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import SegGptForImageSegmentation
from transformers.models.seggpt.modeling_seggpt import SegGptImageSegmentationOutput

class SegGptAdapter(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_path: str = "BAAI/seggpt-vit-large") -> None:
        super(SegGptAdapter, self).__init__()
        self.seggpt = SegGptForImageSegmentation.from_pretrained(model_path)
        # height is actually 2x the image height
        height, width = self.model.config.image_size
        num_channels = self.model.config.num_channels

        # I'm halving the height because the height in config is actually 2x the image height
        # Image prompt is concataned with input image inside the HF implementation
        # Same goes for prompt mask
        self.image_prompt_tensor = nn.Parameter(torch.randn(1, num_channels, height // 2, width))
        self.prompt_mask_tensor = nn.Parameter(torch.randn(1, num_channels, height // 2, width))

        self.freeze_seggpt()
        self.init_weights()
    
    def freeze_seggpt(self) -> None:
        for param in self.seggpt.parameters():
            param.requires_grad = False

    def init_weights(self) -> None:
        nn.init.normal_(self.image_prompt_tensor, std=0.02)
        nn.init.normal_(self.prompt_mask_tensor, std=0.02)

    def forward(
        self, 
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None, 
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        **kwargs
    ) -> SegGptImageSegmentationOutput:
        batch_size = pixel_values.shape[0]

        # Expand image prompt and prompt mask to batch size
        image_prompt_tensor = self.image_prompt_tensor.expand(batch_size, -1, -1, -1)
        prompt_mask_tensor = self.prompt_mask_tensor.expand(batch_size, -1, -1, -1)

        outputs = self.seggpt(
            pixel_values=pixel_values,
            prompt_pixel_values=image_prompt_tensor,
            prompt_masks=prompt_mask_tensor,
            bool_masked_pos=bool_masked_pos,
            labels=labels,
            **kwargs
        )

        return outputs




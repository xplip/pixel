import logging
import math

from transformers import ViTForImageClassification

from ..models.pixel.modeling_pixel import PIXELForPreTraining

logger = logging.getLogger(__name__)


def resize_model_embeddings(model: ViTForImageClassification, max_seq_length: int) -> None:
    """
    Checks whether position embeddings need to be resized. If the specified max_seq_length is longer than
    the model's number of patches per sequence, the position embeddings will be interpolated.
    If max_seq_length is shorter, the position embeddings will be truncated

    Args:
        model (`ViTForImageClassification`):
            The model for which position embeddings may be resized.
        max_seq_length (`int`):
            The maximum sequence length that determines the number of patches (excluding CLS patch) in the
            model.
    """

    patch_size = model.config.patch_size
    if isinstance(model.config.image_size, tuple) or isinstance(model.config.image_size, list):
        old_height, old_width = model.config.image_size
    else:
        old_height, old_width = (model.config.image_size, model.config.image_size)

    # ppr means patches per row (image is patchified into grid of [ppr * ppr])
    old_ppr = math.sqrt(old_height * old_width) // patch_size
    new_ppr = math.sqrt(max_seq_length)

    if old_ppr < new_ppr:
        # Interpolate position embeddings
        logger.info(f"Interpolating position embeddings to {max_seq_length}")
        model.config.interpolate_pos_encoding = True
    elif old_ppr > new_ppr:
        logger.info(f"Truncating position embeddings to {max_seq_length}")
        # Truncate position embeddings
        old_pos_embeds = model.vit.embeddings.position_embeddings[:, : max_seq_length + 1, :]
        model.vit.embeddings.position_embeddings.data = old_pos_embeds.clone()
        # Update image_size
        new_height = int(new_ppr * patch_size) if old_height == old_width else int(patch_size)
        new_width = int(new_ppr * patch_size) if old_height == old_width else int(patch_size * new_ppr ** 2)
        model.config.image_size = [new_height, new_width]
        model.image_size = [new_height, new_width]
        model.vit.embeddings.patch_embeddings.image_size = [new_height, new_width]


def truncate_decoder_pos_embeddings(model: PIXELForPreTraining, max_seq_length: int) -> None:
    """
    Truncates the position embeddings in a PIXEL Decoder

    Args:
        model (`PIXELForPreTraining`):
            The model whose decoder's position embeddings are truncated
        max_seq_length (`int`):
            The maximum sequence length that determines the number of patches (excluding CLS patch) in the
            model.
    """

    logger.info(f"Truncating decoder position embeddings to {max_seq_length}")
    if max_seq_length > model.decoder.decoder_pos_embed.shape[1]:
        model.config.interpolate_pos_encoding = True
    else:
        # Truncate position embeddings
        old_pos_embeds = model.decoder.decoder_pos_embed[:, : max_seq_length + 1, :]
        model.decoder.decoder_pos_embed.data = old_pos_embeds.clone()

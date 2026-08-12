# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib

from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Literal

from ni_lfm.data import modality_transforms as mt
from ni_lfm.models.decoder_embeddings import ImageTokenDecoderEmbedding, SequenceDecoderEmbedding
from ni_lfm.models.encoder_embeddings import (
    ImageEncoderEmbedding,
    ImageTokenEncoderEmbedding,
    SequenceEncoderEmbedding,
)

STATIC_MAPS_FIELDS = (
    "ALBEDO",
    "AVG_ILLUM",
    "DICE",
    "GRAVITY",
    "HPAR",
    "HYDROGEN",
    "ROCK_ABUNDANCE",
    "ROUGHNESS",
    "SP_MINERALOGY_feo",
    "SP_MINERALOGY_high_calcium_pyroxene",
    "SP_MINERALOGY_low_calcium_pyroxene",
    "SP_MINERALOGY_nanophase_iron",
    "SP_MINERALOGY_olivine",
    "SP_MINERALOGY_omat",
    "SP_MINERALOGY_plagioclase",
    "SW_FE_mpfe",
    "SW_FE_npfe",
    "SW_FE_smfe",
    "TBOL_POLES_closest",
    "TBOL_closest",
    "TIO2",
    "TREG",
    "WAC_NORM_REF_321",
    "WAC_NORM_REF_360",
    "WAC_NORM_REF_415",
    "WAC_NORM_REF_566",
    "WAC_NORM_REF_604",
    "WAC_NORM_REF_643",
    "WAC_NORM_REF_689",
)


@dataclass
class ModalityInfoImg:
    """Information for untokenized image modalities.

    Args:
        encoder_embedding: encoder embedding class.
        encoder_kwargs: kwargs for encoder embedding initialization.
        decoder_embedding: decoder embedding class (None for raw images).
        decoder_kwargs: kwargs for decoder embedding initialization.
        type: Modality type ("img").
        pretokenized: whether modality is pretokenized.
        data_range: expected data range.
    """
    encoder_embedding: Any | None = ImageEncoderEmbedding
    encoder_kwargs: dict = field(default_factory=dict)
    decoder_embedding: Any | None = None
    decoder_kwargs: dict = field(default_factory=dict)
    type: Literal["img"] = "img"
    pretokenized: bool = False
    data_range: tuple[float, float] | None = None


@dataclass
class ModalityInfoImgTokenized(ModalityInfoImg):
    """Information for tokenized image modalities."""
    encoder_embedding: Any | None = ImageTokenEncoderEmbedding
    decoder_embedding: Any | None = ImageTokenDecoderEmbedding
    pretokenized: bool = True


@dataclass
class ModalityInfoSeq:
    """Information for sequence modalities."""
    encoder_embedding: type = SequenceEncoderEmbedding
    encoder_kwargs: dict = field(default_factory=dict)
    decoder_embedding: type = SequenceDecoderEmbedding
    decoder_kwargs: dict = field(default_factory=dict)
    type: Literal["seq", "seq_emb", "seq_token"] = "seq"


def _generate_uint15_hash(seed_str):
    """Generates a hash of the seed string as an unsigned int15 integer."""
    return int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % (2**15)


def compute_modality_id(mod_name: str, modality_info: dict) -> int:
    """Compute a unique, deterministic ID for a modality based on its name and key properties.

    Args:
        mod_name: The modality name (e.g., "tok_dtm").
        modality_info: The modality_info dict built at runtime.

    Returns:
        A uint15 hash (0-32767) unique to this modality configuration
    """
    id_components = [mod_name]  # Start with modality name only and add configurable properties

    if modality_info.get("pretokenized") is not None:
        id_components += ["c" + str(modality_info.get("codebook_size", "x")),
                          "n" + str(modality_info.get("num_codebooks", "x"))]

    if modality_info["type"] == "img":
        id_components += ["p" + str(modality_info.get("patch_size", "")),
                          "i" + str(modality_info.get("input_size", "")),
                          str(modality_info.get("tokenizer", "untok"))]

    id_string = "-".join(id_components)
    hash_value = _generate_uint15_hash(id_string)

    return hash_value


MODALITY_INFO = {
    # image modalities - pixel
    "vis": asdict(ModalityInfoImg(data_range=(0.0, 1.0))),
    "uv": asdict(ModalityInfoImg(data_range=(0.0, 1.0))),
    "dtm": asdict(ModalityInfoImg(data_range=(-9500.0, 10800.0))),
    "slope": asdict(ModalityInfoImg(data_range=(0.0, 90.0))),
    "aspect": asdict(ModalityInfoImg(data_range=(-1.0, 1.0))),
    "aspect_3m": asdict(ModalityInfoImg(data_range=(-1.0, 1.0))),
    "dtm_3m": asdict(ModalityInfoImg(data_range=(-9500.0, 10800.0))),
    "nac": asdict(ModalityInfoImg(data_range=(0., 1.))),
    "slope_3m": asdict(ModalityInfoImg(data_range=(0.0, 90.0))),

    # Sequence modalities
    "metadata": asdict(ModalityInfoSeq(
        encoder_kwargs={"vocab_size": 60_757},
        decoder_kwargs={"vocab_size": 60_757},
    )),

    "static_maps": asdict(ModalityInfoSeq(
        encoder_kwargs={"vocab_size": 60_757},
        decoder_kwargs={"vocab_size": 60_757},
    )),

    # Tokenized image modalities
    "tok_vis": asdict(ModalityInfoImgTokenized()),
    "tok_uv": asdict(ModalityInfoImgTokenized()),
    "tok_dtm": asdict(ModalityInfoImgTokenized()),
    "tok_slope": asdict(ModalityInfoImgTokenized()),
    "tok_aspect": asdict(ModalityInfoImgTokenized()),
    "tok_aspect_3m": asdict(ModalityInfoImgTokenized()),
    "tok_dtm_3m": asdict(ModalityInfoImgTokenized()),
    "tok_nac": asdict(ModalityInfoImgTokenized()),
    "tok_slope_3m": asdict(ModalityInfoImgTokenized()),
}

MODALITY_TRANSFORMS = {
    # Lunar untokenized modalities
    "vis": mt.UntokLunarTransform,
    "uv": mt.UntokLunarTransform,
    "dtm": mt.UntokLunarTransform,
    "slope": mt.UntokLunarTransform,
    "aspect": mt.UntokLunarTransform,
    "aspect_3m": mt.UntokLunarTransform,
    "dtm_3m": mt.UntokLunarTransform,
    "nac": mt.UntokLunarTransform,
    "slope_3m": mt.UntokLunarTransform,

    # Text modalities
    "metadata": mt.MetadataTransform,
    "static_maps": partial(mt.StaticMapsTransform, selected_vars=list(STATIC_MAPS_FIELDS)),
}


def setup_modality_transform(domains: list[str], modality_info: dict):
    """Build per-modality transforms for raw modalities."""
    modality_transform = dict()
    for mod in domains:
        info = modality_info[mod]
        if info.get("pretokenized"):
            continue
        if info["type"] == "img":
            stats = info["stats"]
            modality_transform[mod] = MODALITY_TRANSFORMS[mod](
                mean=stats["mean"],
                std=stats["std"],
                min=stats["min"],
                max=stats["max"],
                channels=stats["channels"],
                pre_resize=info["pre_resize"],
                scaler=info.get("scaler"),
            )
        else:
            modality_transform[mod] = MODALITY_TRANSFORMS[mod]()
    return modality_transform

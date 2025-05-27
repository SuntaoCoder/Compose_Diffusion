import os
from pathlib import Path
from typing import Literal

import fire

from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_classes import (
    PRETRAINED_MODEL_NAME,
    MatterGenCheckpointInfo,
)
from mattergen.compose_generator import ComposeCrystalGenerator


def main(
    output_path: str,
    pretrained_name: list[PRETRAINED_MODEL_NAME] | None = None,
    model_path: list[str] | None = None,
    batch_size: int = 64,
    num_batches: int = 1,
    config_overrides: list[str] | None = None,
    checkpoint_epoch: Literal["best", "last"] | int = "last",
    properties_to_condition_on: TargetProperty | None = None,
    sampling_config_path: str | None = None,
    sampling_config_name: str = "combination",
    sampling_config_overrides: list[str] | None = None,
    record_trajectories: bool = True,
    diffusion_guidance_factor: float | None = None,
    strict_checkpoint_loading: bool = True,
    target_compositions: list[dict[str, int]] | None = None,
):

    assert (
        pretrained_name is not None or model_path is not None
    ), "Either pretrained_name or model_path must be provided."

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sampling_config_overrides = sampling_config_overrides or []
    config_overrides = config_overrides or []
    properties_to_condition_on = properties_to_condition_on or {}
    target_compositions = target_compositions or []

    if pretrained_name is not None:
        checkpoint_info_list = [MatterGenCheckpointInfo.from_hf_hub(name, config_overrides=config_overrides) for name in pretrained_name]
    else:
        checkpoint_info_list = [MatterGenCheckpointInfo(
            model_path=Path(path).resolve(),
            load_epoch=checkpoint_epoch,
            config_overrides=config_overrides,
            strict_checkpoint_loading=strict_checkpoint_loading)
            for path in model_path
            ]
        
    _sampling_config_path = Path(sampling_config_path) if sampling_config_path is not None else None
    generator = ComposeCrystalGenerator(
        checkpoint_info=checkpoint_info_list,
        properties_to_condition_on=properties_to_condition_on,
        batch_size=batch_size,
        num_batches=num_batches,
        sampling_config_name=sampling_config_name,
        sampling_config_path=_sampling_config_path,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
        diffusion_guidance_factor=(
            diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
        ),
        target_compositions_dict=target_compositions,
    )
    generator.compose_generate(output_dir=Path(output_path))


def _main():
    fire.Fire(main)


if __name__ == "__main__":
    _main()

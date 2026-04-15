"""Hydra 入口：对 BOP 数据集跑 ISM 测试并写出 npz（与官方 SAM-6D 一致，并支持自定义 split / 输出子目录）。"""

import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="configs", config_name="run_inference")
def run_inference(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg["runtime"]["output_dir"]
    logging.info("Hydra output dir: %s", output_path)
    logging.info("Initializing trainer")

    if cfg.machine.name == "slurm":
        num_gpus = int(os.environ["SLURM_GPUS_ON_NODE"])
        num_nodes = int(os.environ["SLURM_NNODES"])
        cfg.machine.trainer.devices = num_gpus
        cfg.machine.trainer.num_nodes = num_nodes
        logging.info("Slurm: %s gpus, %s nodes", num_gpus, num_nodes)

    trainer = instantiate(cfg.machine.trainer)

    default_ref_dataloader_config = cfg.data.reference_dataloader
    default_query_dataloader_config = cfg.data.query_dataloader

    query_dataloader_config = default_query_dataloader_config.copy()
    ref_dataloader_config = default_ref_dataloader_config.copy()

    user_split = cfg.data.query_dataloader.get("split")
    if user_split is not None and str(user_split).strip() != "":
        query_dataloader_config.split = str(user_split).strip()
    elif cfg.dataset_name in ["hb", "tless"]:
        query_dataloader_config.split = "test_primesense"
    else:
        query_dataloader_config.split = "test"

    query_dataloader_config.root_dir += f"{cfg.dataset_name}"
    query_dataset = instantiate(query_dataloader_config)

    logging.info("Initializing model")
    model = instantiate(cfg.model)

    model.ref_obj_names = cfg.data.datasets[cfg.dataset_name].obj_names
    model.dataset_name = cfg.dataset_name

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=1,
        num_workers=cfg.machine.num_workers,
        shuffle=False,
    )

    if cfg.model.onboarding_config.rendering_type == "pyrender":
        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "pbr":
        logging.info("Using BlenderProc / PBR reference templates")
        ref_dataloader_config._target_ = "provider.bop_pbr.BOPTemplatePBR"
        ref_dataloader_config.root_dir = f"{query_dataloader_config.root_dir}"
        ref_dataloader_config.template_dir += f"templates_pyrender/{cfg.dataset_name}"
        os.makedirs(ref_dataloader_config.template_dir, exist_ok=True)
        ref_dataset = instantiate(ref_dataloader_config)
        ref_dataset.load_processed_metaData(reset_metaData=True)
    else:
        raise NotImplementedError(
            f"rendering_type={cfg.model.onboarding_config.rendering_type}"
        )

    model.ref_dataset = ref_dataset

    subdir = cfg.get("prediction_subdir")
    if subdir is not None and str(subdir).strip() != "":
        model.name_prediction_file = str(subdir).strip()
    else:
        model.name_prediction_file = f"result_{cfg.dataset_name}"

    logging.info(
        "Query split=%s prediction_subdir=%s (under predictions/%s/)",
        query_dataloader_config.split,
        model.name_prediction_file,
        cfg.dataset_name,
    )
    trainer.test(model, dataloaders=query_dataloader)
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()

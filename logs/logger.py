import yaml
import logging
import logging.handlers
import pathlib
from typing import cast

def get_logger(name: str = "pipeline"):
    
    current_dir = pathlib.Path(__file__).parent.resolve() 
    
    project_root = current_dir.parent
    
    cfg_path = project_root / "configs" / "config.yaml"
    
    cfg = cast(dict,
        yaml.safe_load(cfg_path.read_text())
        if cfg_path.is_file() 
        else {}
    )
    #cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.is_file() else {}
    log_cfg = cfg.get("logging", {})

    logger = logging.getLogger(name)
    logger.setLevel(log_cfg.get("level", "INFO"))

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_cfg.get("level", "INFO"))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file handler with rotation
    log_file_rel = log_cfg.get("file", "logs/pipeline.log")
    log_file = project_root / log_file_rel 
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when=log_cfg.get("rotate_when", "midnight"),
        backupCount=log_cfg.get("backup_count", 7),
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
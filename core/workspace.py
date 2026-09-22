"""
Xrexze Workspace Manager.

Handles creating and managing project directories, scaffolding the
folder structure, and reading/writing the project config.json.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


def create_workspace(project_name: str, base_dir: Path) -> Path:
    """
    Scaffolds the project folder structure and writes config.json.

    Parameters
    ----------
    project_name : str
        The name of the new project.
    base_dir : Path
        The root directory where projects are stored (e.g., ./projects).

    Returns
    -------
    Path
        The absolute path to the newly created project root.
    """
    root = base_dir / project_name
    
    # Scaffolding
    (root / "1_Source_Files").mkdir(parents=True, exist_ok=True)
    
    workspace_dir = root / "2_Workspace"
    (workspace_dir / "pages").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "panels").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "audio").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "chunks").mkdir(parents=True, exist_ok=True)
    
    (root / "3_Final_Exports").mkdir(parents=True, exist_ok=True)
    (root / "memory").mkdir(parents=True, exist_ok=True)
    
    # Default Config
    config = {
        "colab_url": "",
        "voice_id": "omnivoice-default",
        "enable_memory": True,
        "created_at": datetime.now().isoformat()
    }
    
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    
    logger.info(f"Created new project workspace at {root}")
    return root


def list_projects(base_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan base_dir for subdirs containing config.json.
    Returns a list of dicts sorted by modified time (newest first).
    """
    projects = []
    
    if not base_dir.exists():
        return projects
        
    for item in base_dir.iterdir():
        if item.is_dir():
            config_path = item / "config.json"
            if config_path.exists():
                try:
                    mtime = config_path.stat().st_mtime
                    dt = datetime.fromtimestamp(mtime)
                    
                    # Read config to verify it's valid
                    config = json.loads(config_path.read_text())
                    
                    projects.append({
                        "name": item.name,
                        "path": item,
                        "last_modified": dt.strftime("%Y-%m-%d %H:%M"),
                        "mtime": mtime,
                        "config": config
                    })
                except Exception as e:
                    logger.warning(f"Failed to read project {item.name}: {e}")
                    
    # Sort by mtime descending (newest first)
    projects.sort(key=lambda x: x["mtime"], reverse=True)
    return projects


def load_project_config(project_root: Path) -> Dict[str, Any]:
    """Load and return config.json from a project root."""
    config_path = project_root / "config.json"
    if not config_path.exists():
        return {}
        
    try:
        return json.loads(config_path.read_text())
    except Exception as e:
        logger.error(f"Failed to load config for {project_root.name}: {e}")
        return {}


def save_project_config(project_root: Path, config: Dict[str, Any]) -> None:
    """Save config.json to a project root."""
    config_path = project_root / "config.json"
    try:
        config_path.write_text(json.dumps(config, indent=2))
        logger.debug(f"Saved config for {project_root.name}")
    except Exception as e:
        logger.error(f"Failed to save config for {project_root.name}: {e}")

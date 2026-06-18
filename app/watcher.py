import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Set

from app import config
from app.pipeline import extract_text, enrich

logger = logging.getLogger(__name__)

# Track files currently being processed to avoid duplicate triggers
_processing_files: Set[str] = set()

async def process_file(filepath: Path):
    """Processes a single file from the watch directory."""
    str_path = str(filepath)
    if str_path in _processing_files:
        return

    _processing_files.add(str_path)
    logger.info(f"Watcher picked up file: {filepath.name}")

    try:
        # 1. Extract text
        text = await extract_text(str_path)

        # 2. Enrich (Gatekeeper, Clean, Embed)
        result = await enrich(text=text, source=f"watch_folder:{filepath.name}")
        logger.info(f"Watcher finished processing {filepath.name}. Result: {result}")

        # 3. Move to processed
        dest_dir = Path(config.WATCH_DIR) / "processed"
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(str_path, dest_dir / filepath.name)

    except Exception as e:
        logger.error(f"Watcher failed to process {filepath.name}: {e}")
        # Move to failed
        dest_dir = Path(config.WATCH_DIR) / "failed"
        os.makedirs(dest_dir, exist_ok=True)
        try:
            shutil.move(str_path, dest_dir / filepath.name)
        except Exception as move_e:
            logger.error(f"Failed to move {filepath.name} to failed directory: {move_e}")
    finally:
        _processing_files.remove(str_path)

async def watch_loop():
    """Background loop monitoring the watch directory."""
    watch_dir = Path(config.WATCH_DIR)

    # Ensure base dirs exist
    os.makedirs(watch_dir, exist_ok=True)
    os.makedirs(watch_dir / "processed", exist_ok=True)
    os.makedirs(watch_dir / "failed", exist_ok=True)

    logger.info(f"Starting directory watcher on {watch_dir}")

    while True:
        try:
            for item in watch_dir.iterdir():
                # Skip directories and already processing files
                if item.is_dir() or str(item) in _processing_files:
                    continue

                # Small delay to ensure file is completely written (primitive lock)
                await asyncio.sleep(1)

                # Start processing as a background task
                asyncio.create_task(process_file(item))

        except Exception as e:
            logger.error(f"Error in watch loop: {e}")

        await asyncio.sleep(5) # Poll every 5 seconds

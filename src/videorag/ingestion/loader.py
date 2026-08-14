"""
loader.py
---------
Loads mock CCTV JSON data and converts records into document dicts
suitable for downstream indexing and retrieval.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"camera", "timestamp", "description"}


class CCTVDataLoader:
    """Loads and validates CCTV JSON records."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> list[dict]:
        """Read a JSON array from *path* and return the validated records.

        Args:
            path: Absolute or relative path to the CCTV JSON file.

        Returns:
            A list of raw record dicts, each guaranteed to contain
            ``camera``, ``timestamp``, and ``description`` fields.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file is not a JSON array or a record is
                missing required fields.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"CCTV data file not found: {path}")

        logger.info("Loading CCTV data from '%s'", path)
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array at top level, got {type(data).__name__}"
            )

        validated: list[dict] = []
        for idx, record in enumerate(data):
            missing = REQUIRED_FIELDS - set(record.keys())
            if missing:
                raise ValueError(
                    f"Record {idx} is missing required fields: {missing}"
                )
            validated.append(record)

        logger.info("Loaded %d CCTV records", len(validated))
        return validated

    def load_as_documents(self, path: str) -> list[dict]:
        """Load records and convert each to a standardised document dict.

        The ``text`` field is formatted as::

            Camera: <camera> | Time: <timestamp> | Event: <description>

        Args:
            path: Path to the CCTV JSON file.

        Returns:
            A list of document dicts with keys:
            ``id``, ``camera``, ``timestamp``, ``description``,
            ``text``, ``metadata``.
        """
        records = self.load(path)
        documents: list[dict] = []

        for idx, record in enumerate(records):
            camera: str = record["camera"]
            timestamp: str = record["timestamp"]
            description: str = record["description"]

            text = (
                f"Camera: {camera} | Time: {timestamp} | Event: {description}"
            )
            metadata = {
                "camera": camera,
                "timestamp": timestamp,
                "image_path": record.get("image_path", ""),
            }

            doc = {
                "id": idx,
                "camera": camera,
                "timestamp": timestamp,
                "description": description,
                "text": text,
                "metadata": metadata,
            }
            documents.append(doc)

        logger.info("Converted %d records to documents", len(documents))
        return documents

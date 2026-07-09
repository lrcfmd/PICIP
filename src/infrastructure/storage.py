from pathlib import Path
from urllib.parse import urlparse

from infrastructure.s3 import S3Client


class StorageService:
    """Service for handling file storage operations (downloads/uploads)."""

    def __init__(self, s3_client: S3Client, logger):
        self.s3 = s3_client
        self.logger = logger

    def download_inputs(
        self, inputs: dict, input_dir: Path
    ) -> dict[str, tuple[Path, str]]:
        """Download URI inputs to local files.

        Args:
            inputs: Dict of field_name -> field_spec where field_spec has "uri" and "mime_type"
            input_dir: Directory to download files to

        Returns:
            Dict of field_name -> (local_path, mime_type)
        """
        downloaded = {}
        input_dir.mkdir(parents=True, exist_ok=True)

        for field_name, field_spec in inputs.items():
            if isinstance(field_spec, dict):
                uri = field_spec.get("uri")
                if not uri and "path" in field_spec:
                    uri = field_spec["path"]
                if uri:
                    mime_type = field_spec.get("mime_type", "application/octet-stream")

                    parsed = urlparse(uri)
                    filename = Path(parsed.path).name or f"{field_name}.bin"
                    local_path = input_dir / filename

                    self.s3.download(uri, local_path)
                    downloaded[field_name] = (local_path, mime_type)
                    self.logger.info("downloaded %s to %s", field_name, str(local_path))

        return downloaded

    def upload_outputs(self, output_dir: Path, output_prefix: str = "") -> None:
        """Upload all files from output directory to S3.

        Args:
            output_dir: Directory containing files to upload
            output_prefix: S3 key prefix for uploaded files
        """
        if not output_dir.exists():
            return

        for file_path in output_dir.iterdir():
            if file_path.is_file():
                prefix = output_prefix.rstrip("/") if output_prefix else ""
                key = (
                    f"{prefix}/{file_path.name}"
                    if prefix
                    else file_path.name
                )
                self.s3.upload(file_path, key)
                self.logger.info("uploaded %s to %s", str(file_path), key)

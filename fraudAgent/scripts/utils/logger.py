"""
Centralized Logging & Experiment Tracking Utility
=================================================
Her script bu utility'yi kullanarak:
- Console + dosyaya log yazar
- Metadata (params, metrics) kaydeder
- Timeline'a otomatik ekler
"""

import logging
import json
from pathlib import Path
from datetime import datetime
import sys


class ExperimentLogger:
    """
    Projenin merkezi logging sistemi

    Özellikler:
    - Console + file logging
    - Experiment metadata tracking (JSON)
    - Human-readable summary
    - Project timeline güncelleme
    """

    def __init__(self, script_name, log_dir=None):
        """
        Args:
            script_name: Hangi script çalışıyor (ör: "build_graph")
            log_dir: Log klasörü (default: project_root/logs)
        """
        self.script_name = script_name
        self.start_time = datetime.now()
        self.run_id = self.start_time.strftime("%Y%m%d_%H%M%S")

        # Paths
        if log_dir is None:
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Run-specific directory
        self.run_dir = self.log_dir / "experiments" / f"{script_name}_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Files
        self.pipeline_log_file = self.log_dir / "pipeline.log"
        self.timeline_file = self.log_dir / "project_timeline.md"
        self.metadata_file = self.run_dir / "metadata.json"
        self.summary_file = self.run_dir / "summary.txt"

        # Setup logging
        self._setup_logging()

        # Initialize metadata
        self.metadata = {
            "script_name": script_name,
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "parameters": {},
            "metrics": {},
            "artifacts": [],
            "status": "running"
        }

        # Log başlangıç
        self.log_header()

    def _setup_logging(self):
        """Logging konfigürasyonu"""
        # Logger oluştur
        self.logger = logging.getLogger(f"fraud_ml.{self.script_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers

        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (pipeline.log - tüm loglar burada)
        file_handler = logging.FileHandler(self.pipeline_log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Run-specific summary file
        summary_handler = logging.FileHandler(self.summary_file, encoding='utf-8')
        summary_handler.setLevel(logging.INFO)
        summary_handler.setFormatter(formatter)
        self.logger.addHandler(summary_handler)

    def log_header(self):
        """Script başlangıç header'ı"""
        self.logger.info("=" * 70)
        self.logger.info(f"SCRIPT: {self.script_name}")
        self.logger.info(f"RUN ID: {self.run_id}")
        self.logger.info(f"START TIME: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 70)

    def info(self, message):
        """Info level log"""
        self.logger.info(message)

    def warning(self, message):
        """Warning level log"""
        self.logger.warning(message)

    def error(self, message):
        """Error level log"""
        self.logger.error(message)
        self.metadata["status"] = "failed"

    def log_parameters(self, params):
        """Parametreleri kaydet"""
        self.metadata["parameters"].update(params)
        self.info(f"Parameters: {json.dumps(params, indent=2)}")

    def log_metrics(self, metrics):
        """Metrikleri kaydet"""
        self.metadata["metrics"].update(metrics)
        self.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    def log_artifact(self, artifact_path, description=""):
        """Artifact'leri kaydet (model, graph, vs.)"""
        artifact_info = {
            "path": str(artifact_path),
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.metadata["artifacts"].append(artifact_info)
        self.info(f"Artifact saved: {artifact_path} ({description})")

    def finalize(self, status="success"):
        """Script bittiğinde çağrılır"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        self.metadata["end_time"] = end_time.isoformat()
        self.metadata["duration_seconds"] = duration
        self.metadata["status"] = status

        # Save metadata JSON
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        # Log footer
        self.logger.info("=" * 70)
        self.logger.info(f"STATUS: {status.upper()}")
        self.logger.info(f"DURATION: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
        self.logger.info(f"METADATA: {self.metadata_file}")
        self.logger.info("=" * 70)

        # Update project timeline
        self._update_timeline()

    def _update_timeline(self):
        """Project timeline.md'yi güncelle"""
        # Timeline dosyası yoksa başlık ekle
        if not self.timeline_file.exists():
            with open(self.timeline_file, 'w', encoding='utf-8') as f:
                f.write("# Fraud ML Service - Project Timeline\n\n")
                f.write("Bu dosya tüm script çalıştırmalarını kronolojik olarak kaydeder.\n\n")
                f.write("---\n\n")

        # Yeni entry ekle
        with open(self.timeline_file, 'a', encoding='utf-8') as f:
            f.write(f"## {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {self.script_name}\n\n")
            f.write(f"**Run ID**: `{self.run_id}`  \n")
            f.write(f"**Status**: {self.metadata['status']}  \n")
            f.write(f"**Duration**: {self.metadata.get('duration_seconds', 0):.2f}s  \n\n")

            if self.metadata['parameters']:
                f.write("**Parameters**:\n")
                for key, val in self.metadata['parameters'].items():
                    f.write(f"- {key}: `{val}`\n")
                f.write("\n")

            if self.metadata['metrics']:
                f.write("**Metrics**:\n")
                for key, val in self.metadata['metrics'].items():
                    f.write(f"- {key}: `{val}`\n")
                f.write("\n")

            if self.metadata['artifacts']:
                f.write("**Artifacts**:\n")
                for art in self.metadata['artifacts']:
                    f.write(f"- {art['description']}: `{art['path']}`\n")
                f.write("\n")

            f.write("---\n\n")


# Helper fonksiyon: Hızlı logger oluşturma
def get_logger(script_name):
    """
    Kolay kullanım için helper

    Usage:
        from utils.logger import get_logger
        logger = get_logger("build_graph")
        logger.info("Starting...")
    """
    return ExperimentLogger(script_name)

"""Schemas for PhantomFlow reproducible evaluation framework."""

from enum import Enum
import time
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from phantomflow.contracts.common import Domain, LanguageCode


class SourcePlatform(str, Enum):
    KAGGLE = "KAGGLE"
    HF = "HF"
    OTHER = "OTHER"


class IntendedUse(str, Enum):
    BUILD = "BUILD"
    EVAL = "EVAL"


class AccessStatus(str, Enum):
    ACQUIRED = "ACQUIRED"
    DOWNLOADING = "DOWNLOADING"
    REQUEST_PENDING = "REQUEST_PENDING"
    LOCAL = "LOCAL"


class DatasetClassification(BaseModel):
    """Rigorous dataset classification model satisfying hackathon data provenance rules."""
    dataset_id: str
    name: str
    source_platform: SourcePlatform
    intended_use: IntendedUse
    license: str
    split: str
    access_status: AccessStatus
    provenance: str
    intended_metric: str
    record_count: int = 0
    description: str = ""
    download_url: Optional[str] = None


class CallEvaluationRecord(BaseModel):
    """Detailed record for a single evaluated audio call or test utterance."""
    call_id: str
    dataset_name: str
    source_type: Literal["REAL_TEST", "BENCHMARK_IMPORT", "SYNTHETIC_MOCK"] = "REAL_TEST"
    domain: Domain = Domain.GENERAL
    scenario: str = ""
    audio_condition: Literal["clean", "noisy_musan", "netem_3g"] = "clean"

    # Transcripts & ASR accuracy
    reference_transcript: str = ""
    hypothesis_transcript: str = ""
    wer: Optional[float] = None

    # Intent & Slot Extraction (NLU)
    expected_intent: str = ""
    actual_intent: str = ""
    intent_correct: bool = True
    expected_slots: Dict[str, Any] = Field(default_factory=dict)
    actual_slots: Dict[str, Any] = Field(default_factory=dict)
    slot_f1: float = 1.0

    # Task completion & Routing
    task_completed: bool = True
    routing_path: str = "fast"

    # GhostBarge Interruption
    interruption_tested: bool = False
    interruption_success: Optional[bool] = None
    interruption_cutoff_ms: Optional[float] = None

    # High-resolution Latency (ms)
    stt_latency_ms: Optional[float] = None
    intent_latency_ms: Optional[float] = None
    task_latency_ms: Optional[float] = None
    tts_latency_ms: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    meets_800ms: bool = False
    meets_600ms: bool = False

    tested_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class EvaluationRun(BaseModel):
    """Complete summary of an evaluation run across a test suite or imported benchmark."""
    run_id: str
    run_name: str
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    is_mock: bool = False
    notes: str = ""

    # Aggregate Metrics
    total_calls: int = 0
    overall_tcr: float = 0.0
    railway_tcr: float = 0.0
    appointment_tcr: float = 0.0

    # End-to-end Latency
    p50_e2e_ms: Optional[float] = None
    p90_e2e_ms: Optional[float] = None
    p95_e2e_ms: Optional[float] = None
    pct_under_800ms: float = 0.0
    pct_under_600ms: float = 0.0

    # Robustness & Interruption
    clean_tcr: float = 0.0
    noisy_tcr: float = 0.0
    interruption_success_rate: Optional[float] = None
    avg_wer: Optional[float] = None

    # Call Records
    records: List[CallEvaluationRecord] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

from pathlib import Path

from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.generation.timeseries_generator import TimeSeriesGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import ComplexityProfile, Modality
from casecrawler.validation.golden_cases import load_golden_summary, summarize_golden_case
from casecrawler.validation.synthetic_validator import SyntheticValidator


FIXTURE_DIR = Path("tests/fixtures/golden_cases")


def test_sepsis_moderate_golden_case_summary_is_stable():
    record = _generate_golden_record(
        GenerationRequest(
            topic="sepsis",
            count=1,
            modalities=_golden_modalities(),
        )
    )

    assert summarize_golden_case(record) == load_golden_summary(
        FIXTURE_DIR / "sepsis_moderate.json"
    )


def test_heart_failure_complex_golden_case_summary_is_stable():
    record = _generate_golden_record(
        GenerationRequest(
            topic="heart failure exacerbation",
            count=1,
            complexity=ComplexityProfile.COMPLEX,
            modalities=_golden_modalities(),
        )
    )

    assert summarize_golden_case(record) == load_golden_summary(
        FIXTURE_DIR / "heart_failure_complex.json"
    )


def _generate_golden_record(req: GenerationRequest):
    record = StructuredGenerator().generate("ds-golden", req, 0)
    record = TimeSeriesGenerator().add_time_series(record)
    record = TextGenerator().add_documents(record)
    validation = SyntheticValidator().validate(record)
    return record.model_copy(update={"validation": validation})


def _golden_modalities() -> list[Modality]:
    return [
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
        Modality.LABS,
        Modality.VITALS,
        Modality.TIME_SERIES,
    ]

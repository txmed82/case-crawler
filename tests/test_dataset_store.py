from casecrawler.models.synthetic import (
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
)
from casecrawler.storage.dataset_store import DatasetStore


def test_dataset_store_round_trips_record(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
    )

    store.save_record(record)

    assert store.get_record("rec-1").record_id == "rec-1"
    assert len(store.list_records(dataset_id="ds-1")) == 1


"""Transfer bundle routing must keep memory-only imports independent."""

from app import transfer


def test_memory_only_bundle_does_not_require_document_path():
    bundle = {
        "format": transfer.FORMAT,
        "version": transfer.VERSION,
        "facts": [{"text": "A preference"}],
        "turns": [],
        "documents": [],
    }
    assert not transfer.bundle_has_documents(bundle)


def test_bundle_with_documents_requires_document_path():
    bundle = {
        "format": transfer.FORMAT,
        "version": transfer.VERSION,
        "facts": [],
        "turns": [],
        "documents": [{"text": "Document text"}],
    }
    assert transfer.bundle_has_documents(bundle)


def test_invalid_bundle_does_not_preempt_validation():
    bundle = {"format": "not-longbrain", "version": transfer.VERSION,
              "documents": [{"text": "Document text"}]}
    assert not transfer.bundle_has_documents(bundle)

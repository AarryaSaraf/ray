"""Tests for ``DataSourceV2.schema_needs_file_sample``.

The hook lets a source tell ``_read_datasource_v2`` that its schema does not
come from the data files. Answering ``False`` skips the driver-side file sample
entirely, which does three things at once: the schema comes from wherever the
source knows it (a catalog), a table that is empty but declared still plans
instead of raising ``no files found``, and no partitioning is guessed from path
names.

Every source in the tree that answers ``True`` is covered by the Parquet
suites. Nothing covers ``False`` -- its first real caller is Iceberg -- so
these tests stand in for it with a stub source, and assert the negative that
matters: with ``False``, listing is never called at all.
"""

import pyarrow as pa
import pytest

from ray.data._internal.datasource_v2.chunkers.file_chunker import WholeFileChunker
from ray.data._internal.datasource_v2.datasource_v2 import (
    DatasourceCategory,
    DataSourceV2,
)
from ray.data._internal.datasource_v2.listing.file_indexer import FileIndexer
from ray.data._internal.datasource_v2.readers.base_reader import Reader
from ray.data._internal.datasource_v2.scanners.scanner import Scanner
from ray.data.read_api import _read_datasource_v2

_SCHEMA = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())])


class _ExplodingIndexer(FileIndexer):
    """An indexer that fails the test if anything lists through it."""

    @property
    def file_chunker(self):
        return WholeFileChunker()

    @property
    def yields_read_units(self) -> bool:
        return True

    def list_files(self, paths, **kwargs):
        raise AssertionError("list_files must not be called on the driver")

    def list_file_infos(self, paths, **kwargs):
        raise AssertionError("list_file_infos must not be called on the driver")


class _EmptyIndexer(_ExplodingIndexer):
    """Lists nothing, without objecting to being asked."""

    def list_files(self, paths, **kwargs):
        return iter([])

    def list_file_infos(self, paths, **kwargs):
        return iter([])


class _NoopReader(Reader):
    def read(self, input_split):
        return iter([])


class _StubScanner(Scanner):
    def __init__(self, schema: pa.Schema):
        self._schema = schema

    def read_schema(self) -> pa.Schema:
        return self._schema

    def create_reader(self) -> Reader:
        return _NoopReader()


class _StubSource(DataSourceV2):
    """A source whose schema is declared, not discovered."""

    def __init__(self, *, needs_sample: bool, indexer: FileIndexer):
        super().__init__(name="Stub", category=DatasourceCategory.DATA_LAKE)
        self._needs_sample = needs_sample
        self._indexer = indexer
        self.partitioning_samples = []

    paths = ["stub://table"]
    filesystem = None
    file_extensions = None

    @property
    def schema_needs_file_sample(self) -> bool:
        return self._needs_sample

    def _get_file_indexer(self) -> FileIndexer:
        return self._indexer

    def infer_schema(self, sample) -> pa.Schema:
        assert sample is None, "a declared schema must not be handed a sample"
        return _SCHEMA

    def resolve_partitioning(self, sample):
        self.partitioning_samples.append(sample)
        return None

    def create_scanner(self, schema, filesystem=None, **options) -> Scanner:
        return _StubScanner(schema)


def test_declared_schema_skips_the_file_sample():
    """``False`` plans a read without listing, and keeps the declared schema."""
    source = _StubSource(needs_sample=False, indexer=_ExplodingIndexer())

    ds = _read_datasource_v2(source)

    # Would raise ``AssertionError`` from ``_ExplodingIndexer`` if the driver
    # had listed anything to get here.
    assert ds.schema().base_schema == _SCHEMA
    # ``resolve_partitioning`` still runs, but with nothing to read keys from.
    assert source.partitioning_samples == [None]


def test_empty_listing_is_only_an_error_when_the_schema_needs_files():
    """``True`` + nothing listed is the ``no files found`` error; ``False`` isn't.

    Same indexer both times: the only difference is who is expected to know the
    schema.
    """
    with pytest.raises(ValueError, match="no files found"):
        _read_datasource_v2(_StubSource(needs_sample=True, indexer=_EmptyIndexer()))

    ds = _read_datasource_v2(_StubSource(needs_sample=False, indexer=_EmptyIndexer()))
    assert ds.schema().base_schema == _SCHEMA


def test_the_hook_is_abstract():
    """Adding a datasource has to answer the question, not inherit a default."""

    class _Forgetful(DataSourceV2):
        def infer_schema(self, sample):
            return _SCHEMA

        def create_scanner(self, schema, filesystem=None, **options):
            raise NotImplementedError

    with pytest.raises(TypeError, match="schema_needs_file_sample"):
        _Forgetful(name="Forgetful", category=DatasourceCategory.DATA_LAKE)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))

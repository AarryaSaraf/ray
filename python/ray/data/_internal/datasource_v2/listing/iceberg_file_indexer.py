"""``FileIndexer`` that lists an Iceberg table by asking PyIceberg to plan it."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional

from ray._common.utils import env_integer
from ray.data._internal.datasource_v2.chunkers.file_chunker import (
    FileChunker,
    WholeFileChunker,
)
from ray.data._internal.datasource_v2.iceberg_manifest import manifest_from_scan_tasks
from ray.data._internal.datasource_v2.listing.file_indexer import FileIndexer, FileInfo
from ray.data._internal.datasource_v2.listing.file_manifest import FileManifest
from ray.data._internal.util import MiB

if TYPE_CHECKING:
    from pyarrow.fs import FileSystem
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.manifest import DataFile
    from pyiceberg.table import FileScanTask, Table

    from ray.data._internal.datasource_v2.listing.file_pruners import FilePruner
    from ray.data.block import BlockColumn
    from ray.data.expressions import Expr

logger = logging.getLogger(__name__)

# Estimated *decoded* bytes per read unit. This sets task granularity only:
# the reader decodes one file at a time, so a bin's size does not bound peak
# memory.
_DEFAULT_BIN_PACKING_BYTES = env_integer(
    "RAY_DATA_ICEBERG_BIN_PACKING_BYTES", 128 * MiB
)

# Placeholder expansion factor from compressed file bytes to decoded Arrow
# bytes. Iceberg metadata records a file's compressed size and its row count,
# and nothing anywhere in it records the uncompressed or decoded size, so there
# is no exact number to read -- and measured expansion spans 1x (incompressible
# int64) to 26x (a low-cardinality string column), so no constant is right
# either. Real estimation is separate work landing with the shared size
# estimator; until then this one knob stands in for it.
_DEFAULT_DECODE_RATIO = env_integer("RAY_DATA_ICEBERG_DECODE_RATIO", 4)


def estimate_decoded_size(data_file: "DataFile") -> int:
    """Guess what ``data_file`` will occupy once decoded into Arrow.

    Deliberately crude -- see :data:`_DEFAULT_DECODE_RATIO`. It exists so that
    bin packing is written against decoded bytes, which is the quantity that
    matters, rather than against compressed bytes, which is merely the one
    Iceberg happens to record. Replacing the guess then changes this function
    and nothing else.
    """
    return data_file.file_size_in_bytes * _DEFAULT_DECODE_RATIO


class IcebergFileIndexer(FileIndexer):
    """Lists an Iceberg table's data files and packs them into read units.

    PyIceberg's ``plan_files()`` replaces the directory walk a file-based
    indexer would do: it reads the table's manifests, prunes partitions and
    files against the row filter, and returns one ``FileScanTask`` per
    surviving data file, already carrying that file's delete files. All of the
    input this needs is metadata, so listing does no data IO.

    The result is emitted as bin-packed read units (``yields_read_units`` is
    ``True``), which is also why this runs in a single ``ListFiles`` task:
    packing has to see the whole file stream, and PyIceberg's planner is
    single-machine anyway.
    """

    def __init__(
        self,
        *,
        table_identifier: str,
        catalog_name: str,
        catalog_kwargs: Dict[str, Any],
        scan_kwargs: Dict[str, Any],
        snapshot_id: Optional[int],
        row_filter: "BooleanExpression",
        max_bin_bytes: Optional[int] = None,
    ):
        self._table_identifier = table_identifier
        self._catalog_name = catalog_name
        self._catalog_kwargs = catalog_kwargs
        self._scan_kwargs = scan_kwargs
        self._snapshot_id = snapshot_id
        self._row_filter = row_filter
        self._max_bin_bytes = (
            max_bin_bytes if max_bin_bytes is not None else _DEFAULT_BIN_PACKING_BYTES
        )

    @property
    def file_chunker(self) -> FileChunker:
        # A read unit is always whole files: an Iceberg scan task names a file,
        # never a byte range or row-group range within one.
        return WholeFileChunker()

    @property
    def yields_read_units(self) -> bool:
        return True

    def list_files(
        self,
        paths: "BlockColumn",
        *,
        filesystem: "FileSystem",
        pruners: Optional[List["FilePruner"]] = None,
        preserve_order: bool = False,
        predicate: Optional["Expr"] = None,
        limit: Optional[int] = None,
        projected_columns: Optional[List[str]] = None,
    ) -> Iterable[FileManifest]:
        """Plan the table and yield each bin as it fills.

        ``paths`` and ``pruners`` are ignored: the table identifier, not a path
        set, says what to list, and which files survive is PyIceberg's decision.
        ``predicate`` is the filter the optimizer pushed onto the read; it is
        ANDed into the scan's row filter, which is what lets partition and file
        pruning happen here instead of on already-listed files. ``limit`` and
        ``projected_columns`` are ignored: stopping the plan early once a
        limit's worth of rows has been listed, and sizing bins by projected
        columns rather than whole files, are both refinements on top of this.
        """
        table = self._load_table()
        batch: List["FileScanTask"] = []
        batch_bytes = 0
        for task in self._plan_files(table, predicate):
            batch.append(task)
            batch_bytes += estimate_decoded_size(task.file)
            if batch_bytes >= self._max_bin_bytes:
                yield manifest_from_scan_tasks(batch, table.metadata)
                batch, batch_bytes = [], 0
        if batch:
            yield manifest_from_scan_tasks(batch, table.metadata)

    def list_file_infos(
        self,
        paths: "BlockColumn",
        *,
        filesystem: "FileSystem",
        pruners: Optional[List["FilePruner"]] = None,
        preserve_order: bool = False,
    ) -> Iterable[FileInfo]:
        """Paths and sizes of the table's data files.

        Only used for driver-side schema sampling, which an Iceberg read skips
        (the schema comes from table metadata), so this exists to satisfy the
        interface.
        """
        for task in self._plan_files(self._load_table(), None):
            yield FileInfo(path=task.file.file_path, size=task.file.file_size_in_bytes)

    def _load_table(self) -> "Table":
        from pyiceberg import catalog as pyi_catalog

        catalog = pyi_catalog.load_catalog(self._catalog_name, **self._catalog_kwargs)
        return catalog.load_table(self._table_identifier)

    def _plan_files(
        self, table: "Table", predicate: Optional["Expr"]
    ) -> Iterator["FileScanTask"]:
        from ray.data._internal.datasource_v2.scanners.iceberg_scanner import (
            combine_row_filter,
        )

        row_filter, residual = combine_row_filter(self._row_filter, predicate)
        assert residual is None, (
            "the scanner only reports a pushed predicate it could translate, "
            f"but this one did not translate: {residual}"
        )
        scan = table.scan(
            row_filter=row_filter,
            snapshot_id=self._snapshot_id,
            **self._scan_kwargs,
        )
        logger.debug(
            "Planning Iceberg scan of %s (snapshot %s, filter %s)",
            self._table_identifier,
            self._snapshot_id,
            row_filter,
        )
        return iter(scan.plan_files())

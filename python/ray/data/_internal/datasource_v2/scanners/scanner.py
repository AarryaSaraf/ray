from abc import ABC, abstractmethod
from typing import Generic

import pyarrow as pa

from ray.data._internal.datasource_v2 import InputSplit
from ray.data._internal.datasource_v2.readers.base_reader import Reader
from ray.util.annotations import DeveloperAPI


@DeveloperAPI
class Scanner(ABC, Generic[InputSplit]):
    """Abstract base class for configured scanners.

    A Scanner represents the logical result of reading data, including applied
    filters, projections, limits, and other pushdown operations. It is an
    immutable abstraction: each push operation returns a new Scanner instance
    via cloning rather than mutation.

    The Scanner is responsible for:
    1. Determining the output schema after all projections
    2. Creating Reader instances configured with all pushdowns

    Splitting the input into parallel work units used to live here as a
    ``plan()`` method. That responsibility now belongs to the listing-side
    pipeline (``ListFiles`` + ``FilePartitioner``); scanners only
    need to answer "what schema?" and "give me a reader."
    """

    @abstractmethod
    def read_schema(self) -> pa.Schema:
        """Return the schema that will be produced by this scanner.

        This reflects the schema after all column pruning has been applied.

        Returns:
            PyArrow Schema describing the output data.
        """
        ...

    def preserves_all_rows(self) -> bool:
        """Whether this scanner returns every row of every file it is given.

        Answers one question: may a row count be taken from file metadata
        instead of by reading the data? (See
        ``ray.data._internal.logical.rules.pushdown_count_files``.)

        **The default is ``False`` because the failure is silent.** A scanner
        that reduces rows -- a pushed predicate, a limit, partition pruning --
        but claims otherwise makes ``count()`` return the *pre-filter* total,
        with no error and no way for a caller to notice. Declining merely costs
        a real read. So override this only to report a condition you check
        directly, and include every row-reducing knob the scanner has,
        **whether it arrived via pushdown or was set at construction** (e.g.
        ``read_iceberg(row_filter=...)``).

        Note the polarity is the opposite of
        :meth:`SupportsFilterPushdown.pushed_predicate`, where under-reporting
        is the safe direction: there it only forgoes listing-time pruning, here
        it corrupts a result.
        """
        return False

    @abstractmethod
    def create_reader(self) -> Reader[InputSplit]:
        """Create a Reader configured for this scanner.

        The returned Reader will have all pushdowns (columns, predicates, limits)
        applied and is ready to execute on workers.

        Returns:
            Configured Reader instance.
        """
        ...

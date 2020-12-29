import copy
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import dump, load
from tqdm import tqdm

from deepform.data.add_features import LABEL_COLS, pq_index_and_dir
from deepform.document import Document
from deepform.logger import logger


@dataclass(frozen=True)
class DocumentStore:
    documents: list

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        for doc in self.documents:
            yield doc

    def __getitem__(self, n):
        """Return the pre-processed tokens for a specified document."""
        return self.documents[n]

    def random_document(self):
        return random.choice(self.documents)

    def sample(self, n=None):
        if n is None:
            n = len(self)
        return DocumentStore(random.sample(self.documents, k=n))

    def split(self, val_percent=0.2):
        """Divide into two DocumentStores, e.g. a training and a validation set."""
        docs_copy = copy.deepcopy(self.documents)
        random.shuffle(docs_copy)
        split_index = int(val_percent * len(self))
        return DocumentStore(docs_copy[:split_index]), DocumentStore(
            docs_copy[split_index:]
        )

    @staticmethod
    def open(index_file, config):
        """Load the documents referenced by `index_file` and apply `config`."""
        index_file = Path(index_file)
        doc_index = pd.read_parquet(index_file)
        logger.info(f"{len(doc_index)} documents in index")

        if not config.pad_windows:
            # Filter out documents that are too short for the curent config.
            doc_index = doc_index[doc_index["length"] >= config.window_len]

        # Filter out documents that don't have a sufficiently high match.
        # doc_index = doc_index[doc_index["best_match"] >= config.target_thresh]
        logger.info(f"After applying config {len(doc_index)} documents are available")

        # Sample down to no more than the requested number of documents.
        num_docs = min(config.len_train, len(doc_index))
        doc_index = doc_index.sample(n=num_docs)

        # Load each of the documents, finishing any necessary feature computation.
        slug_to_doc = caching_doc_getter(index_file, config)
        # docs = concurrent.thread_map(slug_to_doc, doc_index["slug"])

        labels = doc_index[LABEL_COLS.keys()]
        docs = [
            slug_to_doc(slug, labels.loc[slug])
            for slug in tqdm(doc_index.index, desc="Creating docs")
        ]
        docs = [d for d in docs if d != None]  # noqa: E711

        return DocumentStore(docs)


def caching_doc_getter(index_file, config):
    _, pq_root = pq_index_and_dir(index_file)
    if config.use_data_cache:
        cache_root = pq_root.parent / "cache" / cache_master_key(config)
        cache_root.mkdir(parents=True, exist_ok=True)

    def slug_to_doc(slug, labels):
        pq_path = pq_root / f"{slug}.parquet"
        graph_path = pq_root / f"{slug}.graph"
        if config.use_data_cache:
            cache_path = cache_root / f"{slug}.joblib"
            try:
                with open(cache_path, "rb") as infile:
                    return load(infile)
            except FileNotFoundError:
                logger.debug(f"Cache file {cache_path} not found")
        try:
            doc = Document.from_parquet(slug, labels, pq_path, graph_path, config)
        except AssertionError:
            logger.warning(f"No correct answers for {slug}, skipping")
            return None
        if config.use_data_cache:
            with open(cache_path, "wb") as outfile:
                dump(doc, outfile)
            logger.debug(f"Wrote document to cache file {cache_path}")
        return doc

    return slug_to_doc


def cache_master_key(config):
    """Create a string determined by any cache-invalidating config elements."""
    return (
        "str{use_string}_"
        "vocab{vocab_size}_"
        "pg{use_page}_"
        "geom{use_geom}_"
        "amt{use_amount}_"
        "pad{pad_windows}_"
        "len{window_len}"
    ).format(**config)

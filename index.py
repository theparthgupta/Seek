"""
Seek — indexing pipeline.

Usage:
    cocoindex update index.py      # first-time setup + full index
    cocoindex update index.py -L   # live watch mode (re-indexes on file save)
"""
import cocoindex
from config import (
    PROJECTS_PATH,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INCLUDED_PATTERNS,
    EXCLUDED_PATTERNS,
)


@cocoindex.transform_flow()
def code_to_embedding(text: cocoindex.DataSlice[str]):
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(model=EMBED_MODEL)
    )


@cocoindex.flow_def(name="SeekIndex")
def seek_index_flow(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope,
):
    data_scope["files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path=PROJECTS_PATH,
            included_patterns=INCLUDED_PATTERNS,
            excluded_patterns=EXCLUDED_PATTERNS,
        )
    )

    code_embeddings = data_scope.add_collector()

    with data_scope["files"].row() as file:
        file["chunks"] = file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        with file["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(model=EMBED_MODEL)
            )
            code_embeddings.collect(
                filename=file["filename"],
                location=chunk["location"],
                code=chunk["text"],
                embedding=chunk["embedding"],
            )

    code_embeddings.export(
        "code_embeddings",
        cocoindex.targets.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )


if __name__ == "__main__":
    cocoindex.cli.cli()

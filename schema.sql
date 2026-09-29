create extension if not exists vector;

create table if not exists chunks (
    id bigserial primary key,
    corpus text not null,
    strategy text not null,
    unit_id text not null,
    label text not null,
    title text not null,
    part text not null default '',
    citation text not null,
    text text not null,
    url text not null,
    embedding vector(384) not null
);

create table if not exists citations (
    corpus text not null,
    from_unit text not null,
    to_unit text not null,
    primary key (corpus, from_unit, to_unit)
);

create index if not exists chunks_lookup on chunks (corpus, strategy, unit_id);

-- Cosine, because the embeddings are normalised and only direction carries meaning here.
create index if not exists chunks_vector on chunks
    using hnsw (embedding vector_cosine_ops);

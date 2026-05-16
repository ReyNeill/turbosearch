#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container="${TURBOSEARCH_POSTGRES_CONTAINER:-turbosearch-postgres-smoke}"
port="${TURBOSEARCH_POSTGRES_PORT:-55433}"
database="${TURBOSEARCH_POSTGRES_DB:-turbosearch_smoke}"
tmpdir="$(mktemp -d)"

cleanup() {
  rm -rf "${tmpdir}"
  TURBOSEARCH_POSTGRES_CONTAINER="${container}" \
    TURBOSEARCH_POSTGRES_PORT="${port}" \
    TURBOSEARCH_POSTGRES_DB="${database}" \
    "${repo_root}/scripts/postgres-db.sh" down >/dev/null
}
trap cleanup EXIT

TURBOSEARCH_POSTGRES_CONTAINER="${container}" \
  TURBOSEARCH_POSTGRES_PORT="${port}" \
  TURBOSEARCH_POSTGRES_DB="${database}" \
  "${repo_root}/scripts/postgres-db.sh" up >/dev/null

postgres_url="postgres://postgres:postgres@127.0.0.1:${port}/${database}"

docker exec "${container}" psql -U postgres -d "${database}" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker exec "${container}" psql -U postgres -d "${database}" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS psql_bm25s;"
docker exec "${container}" psql -U postgres -d "${database}" -v ON_ERROR_STOP=1 -c "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'psql_bm25s') ORDER BY extname;"

printf '%s\n' \
  'Postgres pgvector enables semantic search over client contracts and renewal clauses.' \
  'The psql_bm25s extension provides exact lexical BM25 hits for names, ids, and symbols.' \
  > "${tmpdir}/postgres-search.txt"

printf '%s\n' \
  'SQLite remains the plug and play local backend for small corpora.' \
  'Hybrid search combines vector evidence and BM25 evidence for multimodal corpora.' \
  > "${tmpdir}/local-search.txt"

pushd "${repo_root}" >/dev/null

env \
  TURBOSEARCH_STORAGE=postgres \
  TURBOSEARCH_POSTGRES_URL="${postgres_url}" \
  TURBOSEARCH_POSTGRES_PGVECTOR=true \
  TURBOSEARCH_POSTGRES_PSQL_BM25S=true \
  TURBOSEARCH_POSTGRES_VECTOR_INDEX=true \
  TURBOSEARCH_EMBEDDING_DIMENSIONS=128 \
  bun run src/cli.ts ingest "${tmpdir}" --storage postgres --embedding hash --ocr none --transcription none

bm25_json="${tmpdir}/bm25.json"
hybrid_json="${tmpdir}/hybrid.json"
exact_json="${tmpdir}/exact.json"

env \
  TURBOSEARCH_STORAGE=postgres \
  TURBOSEARCH_POSTGRES_URL="${postgres_url}" \
  TURBOSEARCH_POSTGRES_PGVECTOR=true \
  TURBOSEARCH_POSTGRES_PSQL_BM25S=true \
  TURBOSEARCH_POSTGRES_VECTOR_INDEX=true \
  TURBOSEARCH_EMBEDDING_DIMENSIONS=128 \
  bun run src/cli.ts search "psql bm25 lexical extension" --storage postgres --embedding hash --backend bm25 --k 2 > "${bm25_json}"

env \
  TURBOSEARCH_STORAGE=postgres \
  TURBOSEARCH_POSTGRES_URL="${postgres_url}" \
  TURBOSEARCH_POSTGRES_PGVECTOR=true \
  TURBOSEARCH_POSTGRES_PSQL_BM25S=true \
  TURBOSEARCH_POSTGRES_VECTOR_INDEX=true \
  TURBOSEARCH_EMBEDDING_DIMENSIONS=128 \
  bun run src/cli.ts search "semantic contracts renewal clauses" --storage postgres --embedding hash --backend exact --k 2 > "${exact_json}"

env \
  TURBOSEARCH_STORAGE=postgres \
  TURBOSEARCH_POSTGRES_URL="${postgres_url}" \
  TURBOSEARCH_POSTGRES_PGVECTOR=true \
  TURBOSEARCH_POSTGRES_PSQL_BM25S=true \
  TURBOSEARCH_POSTGRES_VECTOR_INDEX=true \
  TURBOSEARCH_EMBEDDING_DIMENSIONS=128 \
  bun run src/cli.ts search "semantic bm25 contracts" --storage postgres --embedding hash --backend hybrid --k 2 > "${hybrid_json}"

bun -e '
const [bm25Path, exactPath, hybridPath] = process.argv.slice(1);
const read = async (path) => JSON.parse(await Bun.file(path).text());
const bm25 = await read(bm25Path);
const exact = await read(exactPath);
const hybrid = await read(hybridPath);
for (const [name, payload] of [["bm25", bm25], ["exact", exact], ["hybrid", hybrid]]) {
  if (!payload.ok || !Array.isArray(payload.results) || payload.results.length === 0) {
    throw new Error(`${name} search returned no results`);
  }
}
if (!bm25.results.some((result) => result.text.includes("psql_bm25s"))) {
  throw new Error("BM25 search did not return the psql_bm25s evidence chunk");
}
if (!exact.results.some((result) => result.text.includes("semantic search"))) {
  throw new Error("pgvector exact search did not return the semantic evidence chunk");
}
if (!hybrid.results.some((result) => result.text.includes("BM25") || result.text.includes("semantic search"))) {
  throw new Error("hybrid search did not return expected evidence");
}
console.log(JSON.stringify({
  ok: true,
  bm25Top: bm25.results[0].documentPath,
  exactTop: exact.results[0].documentPath,
  hybridTop: hybrid.results[0].documentPath
}, null, 2));
' "${bm25_json}" "${exact_json}" "${hybrid_json}"

popd >/dev/null

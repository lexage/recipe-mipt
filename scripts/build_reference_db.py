"""Build a reference-only copy of the docs DB for the 'DB filtering' ablation
axis (full docs vs API reference only).

Copies the schema + all libraries, but keeps ONLY the `reference` sections and
their documents/examples. Output: data/docs_database_examples_apiref.db.
Pure SQLite, no model server needed. Idempotent (overwrites the output).
"""
import os
import sqlite3

SRC = "C:/Projects/docs_database_examples.db"
DST = "C:/Projects/recipe-mipt/data/docs_database_examples_apiref.db"

os.makedirs(os.path.dirname(DST), exist_ok=True)
if os.path.exists(DST):
    os.remove(DST)

src = sqlite3.connect(SRC)
dst = sqlite3.connect(DST)

# 1) recreate the schema (tables + indexes) verbatim
for (sql,) in src.execute(
    "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
):
    dst.execute(sql)

dst.commit()
dst.execute("ATTACH DATABASE ? AS s", (SRC,))

# 2) copy libraries (all) and only the reference sections
dst.execute("INSERT INTO libraries SELECT * FROM s.libraries")
dst.execute("INSERT INTO sections SELECT * FROM s.sections WHERE name = 'reference'")

# 3) documents + examples belonging to those reference sections
dst.execute(
    "INSERT INTO documents SELECT * FROM s.documents "
    "WHERE section_id IN (SELECT id FROM s.sections WHERE name = 'reference')"
)
dst.execute(
    "INSERT INTO examples SELECT * FROM s.examples "
    "WHERE doc_id IN (SELECT id FROM s.documents "
    "WHERE section_id IN (SELECT id FROM s.sections WHERE name = 'reference'))"
)
dst.commit()

nd = dst.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
ns = dst.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
ne = dst.execute("SELECT COUNT(*) FROM examples").fetchone()[0]
print(f"built {DST}")
print(f"  sections={ns}  documents={nd}  examples={ne}")

src.close()
dst.close()

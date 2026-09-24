"""Сборка архива поставки pk3.zip из закоммиченного состояния репозитория.

    python3 tools/make_archive.py <корень recipe-mipt> <путь к pk3.zip>

В архив попадает ровно то, что лежит в HEAD (git archive): незакоммиченные
правки, результаты прогонов, индексы, .env и каталоги .git не попадают.
Раскладка — как у поставок соседних компонентов:

    pk3.zip
    ├── recipe-mipt/        исходные файлы компонента, конфиги ПМИ, JSON-корпус, DS-1000
    └── recipe-mipt-eval/   скрипты и программные модули испытаний

Совместим с питоном 3.6 (питон узла alibaba).
"""

import io
import os
import subprocess
import sys
import tarfile
import time
import zipfile

EVAL_DIR = "recipe-mipt-eval"


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    repo, out = os.path.abspath(argv[1]), os.path.abspath(argv[2])
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                          stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                           stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    if dirty:
        print("ВНИМАНИЕ: есть незакоммиченные правки — в архив они не попадут:\n" + dirty,
              file=sys.stderr)
    tar_bytes = subprocess.run(["git", "-C", repo, "archive", "--format=tar", "HEAD"],
                               stdout=subprocess.PIPE, check=True).stdout

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    files = {"recipe-mipt": 0, EVAL_DIR: 0}
    size = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name.startswith(EVAL_DIR + "/"):
                arcname, top = member.name, EVAL_DIR
            else:
                arcname, top = "recipe-mipt/" + member.name, "recipe-mipt"
            info = zipfile.ZipInfo(arcname, date_time=time.localtime(member.mtime)[:6])
            info.external_attr = (member.mode & 0o777) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            data = tar.extractfile(member).read()
            archive.writestr(info, data)
            files[top] += 1
            size += len(data)
        archive.comment = ("pk3: компонент фильтрации и генерации, commit " + head).encode("utf-8")
    os.replace(tmp, out)
    print("архив: {}".format(out))
    print("commit: {}".format(head))
    print("файлов: recipe-mipt {}, {} {}; распакованный объём {:.1f} МБ; архив {:.1f} МБ".format(
        files["recipe-mipt"], EVAL_DIR, files[EVAL_DIR], size / 1e6, os.path.getsize(out) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

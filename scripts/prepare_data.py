"""Download the original ImageNet train tar, verify MD5, unpack safely and resume."""
import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import time
import urllib.request

URL = 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar'
SIZE = 147897477120
MD5 = '1d675b47d978889d74fa0da5fadfb00e'
CHUNK = 256 * 1024**2


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def prepare(root, workers):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'ILSVRC2012_img_train.tar'
    marks = root / '.download_parts'
    marks.mkdir(exist_ok=True)
    fd = os.open(archive, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, SIZE)
    count = (SIZE + CHUNK - 1) // CHUNK

    def download(i):
        marker = marks / str(i)
        if marker.exists():
            return i
        start, end = i * CHUNK, min((i + 1) * CHUNK, SIZE) - 1
        for attempt in range(8):
            try:
                req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=120) as response:
                    assert response.status == 206, 'Server must honor byte ranges'
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{SIZE}'
                    pos = start
                    while block := response.read(1024**2):
                        if pos + len(block) > end + 1:
                            raise ValueError('Response exceeds requested range')
                        view = memoryview(block)
                        while view:
                            n = os.pwrite(fd, view, pos)
                            pos += n
                            view = view[n:]
                    assert pos == end + 1, 'Truncated range'
                os.fsync(fd)
                marker.touch()
                return i
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(2**attempt, 60))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for finished, future in enumerate(concurrent.futures.as_completed(
                    [pool.submit(download, i) for i in range(count)]), 1):
                future.result()
                if finished % 8 == 0 or finished == count:
                    print(f'Download chunks {finished}/{count}', flush=True)
    finally:
        os.close(fd)
    print('Verifying complete archive MD5', flush=True)
    digest = hashlib.md5()
    with archive.open('rb') as stream:
        while block := stream.read(16 * 1024**2):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise RuntimeError('Archive checksum mismatch: do not train; inspect/re-download corrupted parts')
    train = root / 'train'
    train.mkdir(exist_ok=True)
    classes = {}
    with tarfile.open(archive, 'r:') as outer:
        for member in outer:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not re.fullmatch(r'n\d{8}\.tar', name):
                raise ValueError(f'Unexpected archive member: {member.name}')
            cls = name[:-4]
            dest = train / cls
            dest.mkdir(exist_ok=True)
            done = dest / '.complete.json'
            if done.exists():
                classes[cls] = json.loads(done.read_text())['count']
                continue
            with outer.extractfile(member) as stream:
                payload = stream.read()
            num = 0
            with tarfile.open(fileobj=io.BytesIO(payload)) as inner:
                for entry in inner:
                    if not entry.isfile():
                        continue
                    filename = Path(entry.name).name
                    if not re.fullmatch(cls + r'_\d+\.JPEG', filename):
                        raise ValueError(f'Unexpected image: {entry.name}')
                    target = dest / filename
                    if not target.exists() or target.stat().st_size != entry.size:
                        with inner.extractfile(entry) as source:
                            temp = target.with_suffix('.partial')
                            temp.write_bytes(source.read())
                            temp.replace(target)
                    num += 1
            classes[cls] = num
            atomic_json(done, {'count': num})
            print(f'Unpacked {len(classes)}/1000 classes', flush=True)
    assert len(classes) == 1000, len(classes)
    assert sum(classes.values()) == 1281167, sum(classes.values())
    # Validate actual extracted file counts, including resumed directories.
    for cls, expected in classes.items():
        assert len(list((train / cls).glob('*.JPEG'))) == expected, cls
    manifest = {'source': URL, 'archive_bytes': SIZE, 'md5': MD5,
                'images': sum(classes.values()), 'classes': classes,
                'status': 'ready', 'completed_unix': time.time()}
    atomic_json(root / 'manifest.json', manifest)
    print('ImageNet ready: 1,281,167 images, 1000 classes; MD5 verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/imagenet')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    prepare(args.root, args.workers)

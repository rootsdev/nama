import glob as glb
import s3fs

fs = s3fs.S3FileSystem()


def glob(path):
    if path.startswith("s3://"):
        return ["s3://" + filename for filename in fs.glob(path)]
    else:
        return glb.glob(path)


def fopen(path, mode=None, encoding=None):
    if path.startswith("s3://"):
        return fs.open(path, mode=mode, encoding=encoding)
    else:
        return open(path, mode=mode, encoding=encoding)

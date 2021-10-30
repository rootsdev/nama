import glob
import s3fs

fs = s3fs.S3FileSystem()


def glob(path):
    if path.startswith("s3://"):
        return ["s3://" + filename for filename in fs.glob(path)]
    else:
        return glob.glob(path)

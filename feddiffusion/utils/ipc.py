# utils/ipc.py
import io, numpy as np

def pack_arrays(**arrays) -> bytes:
    """把 numpy 数组打包压缩成 bytes，用于通过 multiprocessing.Queue 传输。"""
    bio = io.BytesIO()
    # 统一 dtypes，避免平台差异
    fixed = {k: (v.astype(np.float32) if v.dtype.kind=='f' else v.astype(np.int64))
             for k, v in arrays.items()}
    np.savez_compressed(bio, **fixed)
    return bio.getvalue()

def unpack_arrays(blob: bytes) -> dict:
    """把 bytes 还原成 numpy 数组字典。"""
    bio = io.BytesIO(blob)
    with np.load(bio, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

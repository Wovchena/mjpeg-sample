#!/usr/bin/env python3

import io
import sys
import struct
import numpy as np
from PIL import Image


class LEWriter:
    """Little-endian binary file writer with patch support."""
    def __init__(self, path: str):
        # wb+ allows writing and seeking for later patching
        self.f = open(path, 'wb+')

    def tell(self) -> int:
        return self.f.tell()

    def seek(self, pos: int):
        self.f.seek(pos, 0)

    def write(self, b: bytes):
        self.f.write(b)

    def write_u16(self, v: int):
        self.f.write(struct.pack('<H', v))

    def write_u32(self, v: int):
        self.f.write(struct.pack('<I', v))

    def write_i32(self, v: int):
        self.f.write(struct.pack('<i', v))

    def write_fourcc(self, tag: str):
        assert len(tag) == 4
        self.f.write(tag.encode('ascii'))

    def patch_u32(self, pos: int, v: int):
        cur = self.tell()
        self.seek(pos)
        self.write_u32(v)
        self.seek(cur)

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()


class AVIMJPEGWriter:
    """
    Minimal AVI (RIFF) writer for MJPEG streams.
    Writes:
      RIFF('AVI ')
        LIST('hdrl')
          'avih'
          LIST('strl')
            'strh'
            'strf'
        LIST('movi')
          '00dc' ...   (per-frame JPEG)
        'idx1'
    """
    def __init__(self, path: str, width: int, height: int, fps: int):
        self.w = int(width)
        self.h = int(height)
        self.fps = int(fps)
        self.writer = LEWriter(path)

        # Patch locations / offsets
        self.riff_start = 0
        self.riff_size_pos = 0

        self.hdrl_list_start = 0
        self.hdrl_list_size_pos = 0
        self.hdrl_end = 0

        self.avih_data_pos = 0
        self.strh_data_pos = 0

        self.movi_list_start = 0
        self.movi_list_size_pos = 0
        self.movi_data_start = 0

        self.index = []   # list of (offset, size)
        self.frame_count = 0
        self.max_chunk = 0
        self.finalized = False

        self._begin_file()

    def _begin_file(self):
        w = self.writer

        # RIFF header
        self.riff_start = w.tell()
        w.write_fourcc('RIFF')
        self.riff_size_pos = w.tell()
        w.write_u32(0)                  # placeholder for RIFF size
        w.write_fourcc('AVI ')

        # LIST 'hdrl'
        self.hdrl_list_start = w.tell()
        w.write_fourcc('LIST')
        self.hdrl_list_size_pos = w.tell()
        w.write_u32(0)                  # placeholder
        w.write_fourcc('hdrl')

        # 'avih' (MainAVIHeader) - 56 bytes
        w.write_fourcc('avih')
        w.write_u32(56)
        self.avih_data_pos = w.tell()

        usec_per_frame = int(1_000_000 // self.fps)
        w.write_u32(usec_per_frame)     # dwMicroSecPerFrame
        w.write_u32(0)                  # dwMaxBytesPerSec (patch later)
        w.write_u32(0)                  # dwPaddingGranularity
        w.write_u32(0x00000010)         # dwFlags (AVIF_HASINDEX)
        w.write_u32(0)                  # dwTotalFrames (patch later)
        w.write_u32(0)                  # dwInitialFrames
        w.write_u32(1)                  # dwStreams
        w.write_u32(0)                  # dwSuggestedBufferSize (patch later)
        w.write_u32(self.w)             # dwWidth
        w.write_u32(self.h)             # dwHeight
        # dwReserved[4]
        w.write_u32(0); w.write_u32(0); w.write_u32(0); w.write_u32(0)

        # LIST 'strl'
        strl_list_start = w.tell()
        w.write_fourcc('LIST')
        strl_list_size_pos = w.tell()
        w.write_u32(0)                  # placeholder
        w.write_fourcc('strl')

        # 'strh' (AVISTREAMHEADER) - 56 bytes
        w.write_fourcc('strh')
        w.write_u32(56)
        self.strh_data_pos = w.tell()

        w.write_fourcc('vids')          # fccType
        w.write_fourcc('MJPG')          # fccHandler
        w.write_u32(0)                  # dwFlags
        w.write_u16(0)                  # wPriority
        w.write_u16(0)                  # wLanguage
        w.write_u32(0)                  # dwInitialFrames
        w.write_u32(1)                  # dwScale
        w.write_u32(self.fps)           # dwRate
        w.write_u32(0)                  # dwStart
        w.write_u32(0)                  # dwLength (patch later)
        w.write_u32(0)                  # dwSuggestedBufferSize (patch later)
        w.write_u32(0xFFFFFFFF)         # dwQuality
        w.write_u32(0)                  # dwSampleSize (0 = variable)

        # rcFrame (left, top, right, bottom) as 16-bit each
        w.write_u16(0)                  # left
        w.write_u16(0)                  # top
        w.write_u16(self.w & 0xFFFF)    # right
        w.write_u16(self.h & 0xFFFF)    # bottom

        # 'strf' (BITMAPINFOHEADER) - 40 bytes
        w.write_fourcc('strf')
        w.write_u32(40)                 # chunk size
        w.write_u32(40)                 # biSize
        w.write_i32(self.w)             # biWidth
        w.write_i32(self.h)             # biHeight
        w.write_u16(1)                  # biPlanes
        w.write_u16(24)                 # biBitCount
        w.write_fourcc('MJPG')          # biCompression
        w.write_u32(0)                  # biSizeImage (can be 0 for MJPG)
        w.write_i32(0)                  # biXPelsPerMeter
        w.write_i32(0)                  # biYPelsPerMeter
        w.write_u32(0)                  # biClrUsed
        w.write_u32(0)                  # biClrImportant

        # Patch LIST 'strl' size
        strl_end = w.tell()
        strl_size = strl_end - (strl_list_start + 8)
        w.patch_u32(strl_list_size_pos, strl_size)

        # Mark end of hdrl for later size patch
        self.hdrl_end = w.tell()

        # LIST 'movi'
        self.movi_list_start = w.tell()
        w.write_fourcc('LIST')
        self.movi_list_size_pos = w.tell()
        w.write_u32(0)                  # placeholder
        w.write_fourcc('movi')
        self.movi_data_start = w.tell() # position after "movi" tag

    def add_rgb_frame_as_jpeg(self, rgb: np.ndarray, quality: int = 85):
        """
        Append one frame:
          - rgb: numpy array (H, W, 3), dtype=uint8
          - encoded to JPEG in-memory (no temp files)
        """
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be a (H, W, 3) uint8 array")

        # Encode RGB -> JPEG in-memory with Pillow
        img = Image.fromarray(rgb, mode='RGB')
        bio = io.BytesIO()
        # You can tweak subsampling/quality if desired
        img.save(bio, format='JPEG', quality=quality)
        jpeg_bytes = bio.getvalue()

        w = self.writer

        # Write '00dc' chunk: FOURCC, size, data, padding
        chunk_start = w.tell()
        w.write_fourcc('00dc')
        w.write_u32(len(jpeg_bytes))
        w.write(jpeg_bytes)
        if (len(jpeg_bytes) & 1) != 0:
            w.write(b'\x00')  # word-align

        # Index offset -> start of the CHUNK HEADER (00dc), relative to movi DATA star
        offset_from_movi_data = chunk_start - self.movi_data_start
        self.index.append((offset_from_movi_data, len(jpeg_bytes)))

        self.frame_count += 1
        if len(jpeg_bytes) > self.max_chunk:
            self.max_chunk = len(jpeg_bytes)

    def finish(self):
        if self.finalized:
            return
        w = self.writer

        # Close 'movi' list (patch size)
        movi_end = w.tell()
        movi_size = movi_end - (self.movi_list_start + 8)
        w.patch_u32(self.movi_list_size_pos, movi_size)

        # Write legacy 'idx1' index
        w.write_fourcc('idx1')
        w.write_u32(len(self.index) * 16)
        for (ofs, size) in self.index:
            w.write_fourcc('00dc')     # dwChunkId
            w.write_u32(0x00000010)    # dwFlags (AVIIF_KEYFRAME)
            w.write_u32(ofs)           # dwOffset (from 'movi' data start, to DATA)
            w.write_u32(size)          # dwSize (data only)

        file_end = w.tell()

        # Patch hdrl list size
        hdrl_size = self.hdrl_end - (self.hdrl_list_start + 8)
        w.patch_u32(self.hdrl_list_size_pos, hdrl_size)

        # Patch RIFF size
        riff_size = file_end - (self.riff_start + 8)
        w.patch_u32(self.riff_size_pos, riff_size)

        # Patch avih fields
        w.patch_u32(self.avih_data_pos + 16, self.frame_count)   # dwTotalFrames
        w.patch_u32(self.avih_data_pos + 28, self.max_chunk)     # dwSuggestedBufferSize
        w.patch_u32(self.avih_data_pos +  4, self.fps * self.max_chunk)  # dwMaxBytesPerSec (rough)

        # Patch strh fields (offsets from start of strh data)
        w.patch_u32(self.strh_data_pos + 32, self.frame_count)   # dwLength (frames)
        w.patch_u32(self.strh_data_pos + 36, self.max_chunk)     # dwSuggestedBufferSize

        self.finalized = True
        w.close()


def generate_random_frames(width: int, height: int, fps: int, seconds: int, seed: int = 42):
    """
    Generator yielding random RGB frames as numpy arrays (H, W, 3), dtype=uint8.
    """
    total = fps * seconds
    rng = np.random.default_rng(seed)
    for _ in range(total):
        # Integers in [0, 255], uint8
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        yield frame


def main(argv=None):
    argv = argv or sys.argv
    # Defaults
    out_path = "random_mjpeg.avi"
    width = 640
    height = 360
    fps = 30
    seconds = 2
    quality = 85

    # CLI: script [out.avi] [width] [height] [fps] [seconds] [quality]
    if len(argv) >= 2: out_path = argv[1]
    if len(argv) >= 3: width = int(argv[2])
    if len(argv) >= 4: height = int(argv[3])
    if len(argv) >= 5: fps = int(argv[4])
    if len(argv) >= 6: seconds = int(argv[5])
    if len(argv) >= 7: quality = int(argv[6])

    avi = AVIMJPEGWriter(out_path, width, height, fps)

    try:
        for frame in generate_random_frames(width, height, fps, seconds, seed=42):
            avi.add_rgb_frame_as_jpeg(frame, quality=quality)
        avi.finish()
    except Exception:
        # Ensure file handle is closed on error
        try:
            avi.writer.close()
        finally:
            raise

    total_frames = fps * seconds
    print(f"Wrote {out_path} ({total_frames} frames, {width}x{height} @ {fps} fps, quality {quality})")


if __name__ == "__main__":
    main()

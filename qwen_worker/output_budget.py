"""Encoded-size bounds retained for offline analysis, not generation routing.

Only metadata-free, 8-bit RGB(A) PNGs from the pinned pipeline/encoder have a
bound here. Unknown formats/encoders have no claimed bound. The service selects
its generation route from memory budgets and fits the actual response afterward.
"""

from PIL import ImageFile, __version__ as pillow_version


def png_byte_bound(width: int, height: int) -> int | None:
    if pillow_version != "12.3.0" or ImageFile.MAXBLOCK != 65536:
        return None
    # At most four 8-bit channels plus a filter byte per scanline. Pillow uses
    # deflateInit2(windowBits=15, memLevel=9, Z_FILTERED), with no intermediate
    # flushes. The general zlib/zlib-ng deflateBound envelope is larger than
    # the tighter default-settings bound (which must not be used at memLevel=9).
    source = height * (4 * width + 1)
    compressed = source + (source + 7) // 8 + (source + 63) // 64 + 11
    # Signature, IHDR, IEND, plus 12 bytes per IDAT chunk. ImageFile._save uses
    # max(MAXBLOCK, width*4) output buffers, emitting full buffers until finish.
    chunk_size = max(ImageFile.MAXBLOCK, width * 4)
    chunks = (compressed + chunk_size - 1) // chunk_size
    return 8 + 25 + 12 + compressed + chunks * 12

import struct
import zlib
import cv2
from PIL import Image
import numpy as np


def create_png_signature():
    return b'\x89PNG\r\n\x1a\n'


def create_ihdr_chunk(width, height):
    chunk_type = b'IHDR'
    data = struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0)  # ! - big-endian, 2I - 2 unsigned int, 5B - 5 unsigned bytes
    return create_chunk(chunk_type, data)


def create_idat_chunk(image_data, compression_level):
    compressor = zlib.compressobj(compression_level)
    compressed_data = compressor.compress(image_data) + compressor.flush()
    return create_chunk(b'IDAT', compressed_data)


def create_iend_chunk():
    return create_chunk(b'IEND', b'')


def create_chunk(chunk_type, data):
    chunk_length = len(data)
    chunk = struct.pack("!I", chunk_length) + chunk_type + data
    crc = zlib.crc32(chunk_type + data) & 0xffffffff
    chunk += struct.pack("!I", crc)
    return chunk


def filter_scanline(scanline, prev_scanline, filter_type):
    if filter_type == 0:  # None
        return scanline
    elif filter_type == 1:  # Sub
        return bytearray((scanline[i] - (scanline[i - 3] if i >= 3 else 0)) & 0xFF for i in range(len(scanline)))
    elif filter_type == 2:  # Up
        return bytearray((scanline[i] - (prev_scanline[i] if prev_scanline else 0)) & 0xFF for i in range(len(scanline)))
    elif filter_type == 3:  # Average
        return bytearray(
            (scanline[i] - ((scanline[i - 3] if i >= 3 else 0) + (prev_scanline[i] if prev_scanline else 0)) // 2) & 0xFF
            for i in range(len(scanline)))
    elif filter_type == 4:  # Paeth
        def paeth_predictor(a, b, c):
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            return a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)

        return bytearray(
            (scanline[i] - paeth_predictor(scanline[i - 3] if i >= 3 else 0, prev_scanline[i] if prev_scanline else 0,
                                           prev_scanline[i - 3] if (i >= 3 and prev_scanline) else 0)) & 0xFF for i in
            range(len(scanline)))


def apply_filter(image, filter_type):
    height, width, _ = image.shape
    scanlines = []
    prev_scanline = None

    for y in range(height):
        scanline = image[y, :, :3].tobytes()
        filtered_scanline = filter_scanline(scanline, prev_scanline, filter_type)
        scanlines.append(bytes([filter_type]) + filtered_scanline)
        prev_scanline = scanline

    return b''.join(scanlines)


def save_as_png(image, filename, filter_type, compression_level):
    height, width, _ = image.shape
    png_data = create_png_signature()
    png_data += create_ihdr_chunk(width, height)

    scanlines = apply_filter(image, filter_type)

    png_data += create_idat_chunk(scanlines, compression_level)
    png_data += create_iend_chunk()

    with open(filename, 'wb') as f:
        f.write(png_data)


def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_image_from_disk(path):
    image = Image.open(path)
    return np.array(image)


if __name__ == "__main__":
    # image = capture_image_from_camera()
    image = load_image_from_disk('img/featuredimag.png')
    save_as_png(image, 'new_img/output.png', 0, 9)

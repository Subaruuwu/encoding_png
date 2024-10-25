from src.chunk_handling import create_png_signature, create_ihdr_chunk, create_idat_chunk, create_iend_chunk
from src.filters import apply_filter
from src.io_operations import save_file
from app.labels import config_text
from src.file_utils import get_file_size, calculate_compression_ratio
from data.data_file import file_patch
import os

# def save_many_images(list_of_images):
#     for image in list_of_images:
#             save_as_png(load_image_from_disk(file_patch.input_patch),
#                     get_new_patch(file_patch.input_patch,
#                                   file_patch.output_patch),
#                     inverted_dict_of_filters[combo_filters.get()],
#                     int(combo_compression.get()), status_label)
def save_as_png(image, filename, filter_type, compression_level, text_label):
    label = text_label
    config_text(label, 'Конвертация началась\nЭто может занять некоторое\n время\nПожалйста подождите')
    if filter_type == 5:
        res = []
        if len(image.shape) > 2:
            height, width, _ = image.shape
        else:
            height, width = image.shape
        for filter_iter in range(0, 5):
            png_data = create_png_signature()
            png_data += create_ihdr_chunk(width, height)
            scanlines = apply_filter(image, filter_iter)
            png_data += create_idat_chunk(scanlines, compression_level)
            png_data += create_iend_chunk()
            curr_file = save_file(filename, png_data)
            res.append(os.path.getsize(curr_file))
            print(res)
        min_weight = min(res)
        best_filter = res.index(min_weight)

        png_data = create_png_signature()
        png_data += create_ihdr_chunk(width, height)
        scanlines = apply_filter(image, best_filter)
        png_data += create_idat_chunk(scanlines, compression_level)
        png_data += create_iend_chunk()
        save_file(filename, png_data)

    else:
        if len(image.shape) > 2:
            height, width, _ = image.shape
        else:
            height, width = image.shape
        png_data = create_png_signature()
        png_data += create_ihdr_chunk(width, height)
        scanlines = apply_filter(image, filter_type)
        png_data += create_idat_chunk(scanlines, compression_level)
        png_data += create_iend_chunk()
        save_file(filename, png_data)
    print('Done')
    config_text(label, f'Конвертация завершена\nРазмер нового файла - {get_file_size(file_patch.new_image_patch)} Мб\nИзображение сжато по сравнению исходным на {calculate_compression_ratio(file_patch.input_patch, file_patch.new_image_patch) * 100}%')
    return filename


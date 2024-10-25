from app.window import make_window
from app.buttons import make_button, make_combobox
from data.data_file import dict_of_filters, levels_of_compression, inverted_dict_of_filters
from app.labels import make_label, make_text_label
from app.get_path import select_input_files
from data.data_file import file_patch
from src.png_algorithm import save_as_png
from src.io_operations import load_image_from_disk
from src.name_getter import get_new_patch


def start_app():
    window = make_window('PNG-converter')

    text_label = make_label(window, 'Выберите путь к изображению: ',  0, 0, 10, 10)

    combo_filters = make_combobox(dict_of_filters, window, 1, 1, 50, 50)
    combo_compression = make_combobox(levels_of_compression, window, 2, 1, 50, 50)

    label_filter = make_label(window, 'Выберите фильтр', 1,  0, 20, 20)
    label_compression = make_label(window, 'Выберите уровень сжатия', 2,  0, 20, 20)

    info_label = make_text_label(50, 3, 0, 10, state='disabled')

    path_button = make_button(window, 'Обзор', lambda: select_input_files(start_button, info_label, file_patch), 0, 1, 10, 10)

    status_label = make_label(window, '', 3, 1, 10, 10)

    start_button = make_button(window, 'Начать', lambda: save_as_png(load_image_from_disk(file_patch.input_patch), get_new_patch(file_patch.input_patch, file_patch.output_patch), inverted_dict_of_filters[combo_filters.get()], int(combo_compression.get()), status_label), 4, 0, 0, 10, state='disabled')

    start_button = make_button(window, 'Сжать папку', lambda: save_as_png(load_image_from_disk(file_patch.input_patch),
                                                                     get_new_patch(file_patch.input_patch,
                                                                                   file_patch.output_patch),
                                                                     inverted_dict_of_filters[combo_filters.get()],
                                                                     int(combo_compression.get()), status_label), 4, 0,
                               0, 10, state='disabled')

    window.mainloop()

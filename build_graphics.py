import random
import time
import os
import matplotlib.pyplot as plt

from src.png_algorithm import save_as_png
from src.io_operations import load_image_from_disk

arr_photos = ['8k_state_of_liberty.png']

def get_time_func(func):
    def wrapper(*args):
        t0 = time.time()
        func(args)
        t1= time.time()
        return t1-t0
    return wrapper



def save_png_with_decorator(image, output_file, filter_type, compression_level):
    save_as_png(image, filename=output_file, filter_type=filter_type, compression_level=compression_level)

# image = load_image_from_disk('letter_r.png')
# for i in range(0, 5):
#     new_png = save_as_png(image, 'letter_r_back.png', i, 9)
#     file_size = os.path.getsize(new_png)
#     print(f'Размер файла при фильтре = {i}: {round(file_size/1024, 3)} Мбайт')


def show_table_filters(arr_photos):
    print(''.center(20), end='')
    filters = ['Origin','None', 'Sub', 'Up', 'Average', 'Paeth']
    print(*[str(name).center(20) for name in filters])

    for path in arr_photos:
        image = load_image_from_disk(path)
        print(str(path).center(20), end='|')
        print(str(round(os.path.getsize(path)/1024/1024, 3)).center(20), end='|')
        for k in range(0, 5):
            curr_file = save_as_png(image, filename='curr_out.png', filter_type=k, compression_level=9)
            print((str(round(os.path.getsize(curr_file)/1024/1024, 3)) + ' (' + str(round(os.path.getsize(curr_file)/os.path.getsize(path)*100, 2)) + '%)').center(20), end='|')
        print()


def show_table_degree_compression(arr_photos, filter=4):
    print(''.center(20), end='')
    degrees = ['Origin', 1,2,3,4,5,6,7,8,9]
    print(*[str(name).center(20) for name in degrees])
    dct = {}

    for path in arr_photos:
        image = load_image_from_disk(path)
        print(str(path).center(20), end='|')
        print(str(round(os.path.getsize(path) / 1024 / 1024, 3)).center(20), end='|')

        arr = [round(os.path.getsize(path)/1024/1024, 2)]
        for k in list(range(1, 10)):
            curr_file = save_as_png(image, filename='curr_out.png', filter_type=filter, compression_level=k)
            arr.append(round(os.path.getsize(curr_file) / 1024 / 1024, 2))
            print((str(round(os.path.getsize(curr_file) / 1024 / 1024, 3)) + ' (' + str(
                round(os.path.getsize(curr_file) / os.path.getsize(path) * 100, 2)) + '%)').center(20), end='|')
        print()
        dct[path] = arr

    plt.figure(figsize=(7,7))

    plt.title('Изменение веса изображения от степени сжатия')
    plt.xlabel('Степень сжатия')
    plt.ylabel('Размер изображения, Мбайт')
    for path in dct:
        color = (random.random(),random.random(),random.random())
        plt.plot(list(range(0, 10)), [os.path.getsize(path)/1024/1024]*10, linewidth=1, color=color)
        plt.plot(list(range(0,10)), dct[path], 'o-', label=path, color=color)

    plt.grid(True)
    plt.legend()
    plt.show()


def get_one_graph_info(path: str, compare_raw = False) -> dict:
    degrees = ['Origin', 1, 2, 3, 4, 5, 6, 7, 8, 9]
    filters_name = ['None', 'Sub', 'Up', 'Average', 'Paeth']

    print(path.center(20), end='')
    print(*[str(name).center(20) for name in degrees])

    dct_info = {}

    image = load_image_from_disk(path)
    for filter_id, filter_name in enumerate(filters_name):
        curr_filter_info = [round(os.path.getsize(path)/1024/1024, 2)]
        print(filter_name.center(20), end='|')

        raw_image = load_image_from_disk(path)
        raw_weight = os.path.getsize(raw_image)

        print(str(round(os.path.getsize(path) / 1024 / 1024, 3)).center(20), end='|')  # Столбец Origin

        for degree in degrees[1:]:
            curr_file = save_as_png(image, filename='curr_out.png', filter_type=filter_id, compression_level=degree)
            curr_filter_info.append(round(os.path.getsize(curr_file) / 1024 / 1024, 2))
            print((str(round(os.path.getsize(curr_file) / 1024 / 1024, 3)) +
            ' (' + str(round(os.path.getsize(curr_file) / os.path.getsize(path) * 100, 2)) + '%)').center(20), end='|')
        print()
        dct_info[filter_name] = curr_filter_info

    return dct_info

def show_one_graph_info(dct_info: dict, file_name = ''):

    plt.title(f'Изменение веса изображения {file_name} от выбранного алгоритма и степени сжатия')
    plt.xlabel('Степень сжатия')
    plt.ylabel('Размер изображения, Мбайт')
    marker_lines = ['o-', '1-', '^-', 's-', '*-']
    i = 0
    for filter_name in dct_info:
        plt.plot(list(range(0, 10)), dct_info[filter_name], marker_lines[i], label=filter_name, linewidth=6-i)
        i += 1
    plt.grid(True)
    plt.legend()
    plt.show()


def show_time_quality_relationship(arr_photo: list, compression_level=1):
    pxl_sizes = []
    min_image = load_image_from_disk(arr_photo[0])
    min_amount = min_image.shape[0]*min_image.shape[1]
    for path_image in arr_photo: # получение размеров изображений
        image = load_image_from_disk(path_image)
        pxl_sizes.append(f'{image.shape[1]}x{image.shape[0]}\n({round(image.shape[0]*image.shape[1]/min_amount, 1)}X)')

    time_results = []
    filters = ['None', 'Sub', 'Up', 'Average', 'Paeth']
    for filter_id, filter_name in enumerate(filters):
        curr_filter_results = []
        for path_image in arr_photo:
            image = load_image_from_disk(path_image)
            t0 = time.time()
            save_as_png(image, 'curr_out.png', filter_type=filter_id, compression_level=compression_level)
            t1 = time.time()
            curr_filter_results.append(round(t1-t0,3))
        print(curr_filter_results)
        time_results.append(curr_filter_results)

    plt.title(f'Зависимость времени конвертации от размерности изображения при уровне сжатия = {compression_level}')
    plt.xlabel('Размерность изображения, pxl')
    plt.ylabel('Затраченное время, с')
    for i, filter_name in enumerate(filters):
        plt.plot(pxl_sizes, time_results[i], 'o-', label=filter_name, linewidth=2)

    plt.grid(True)
    plt.legend()
    plt.show()


# show_table_filters(arr_photos)
# show_table_degree_compression(arr2_photos, filter=4)

a = get_one_graph_info('img/salade-coloriage-nice-coloriage-salade-a-imprimer-sur-coloriages-fo-de-salade-coloriage.png')
show_one_graph_info(a, '2,5k_black.png')
show_time_quality_relationship(['img/hd.png', 'img/two_k.png', 'img/four_k.png'], compression_level=9)

# a = get_one_graph_info('img/state_of_liberty/4k_state_of_liberty.png')
# show_one_graph_info(a, '4k_state_of_liberty.png')g
show_time_quality_relationship([
                                'img/state_of_liberty/l2715x1527_state_of_liberty.png',
                                'img/state_of_liberty/l3840x2160_state_of_liberty.png',
                                'img/state_of_liberty/l4703x2745_state_of_liberty.png',
                                'img/state_of_liberty/l5431x3055_state_of_liberty.png',
                                'img/state_of_liberty/l6072x3415_state_of_liberty.png',
                                'img/state_of_liberty/l6651x3741_state_of_liberty.png',
                                'img/state_of_liberty/l7184x4041_state_of_liberty.png',
                                'img/state_of_liberty/8k_state_of_liberty.png'], compression_level=9)


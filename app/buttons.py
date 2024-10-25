import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style

def make_button(window, text, command, row, column, x, y, state='normal'):
    button = ttk.Button(window, text=text, command=command, state=state)
    button.grid(row=row, column=column, padx=x, pady=y)
    return button


def make_combobox(dict_of_tasks: dict, window: tk, row: int, column: int, x: int, y: int):
    combobox = ttk.Combobox(window, values=[task for task in dict_of_tasks.values()])
    combobox.grid(row=row, column=column, padx=x, pady=y)
    combobox.bind("<<ComboboxSelected>>")
    return combobox


def make_button(window, text, command, row, column, x, y, state='normal'):
    button = ttk.Button(window, text=text, command=command, state=state)
    button.grid(row=row, column=column, padx=x, pady=y)
    return button


def make_combobox(dict_of_tasks: dict, window: tk, row: int, column: int, x: int, y: int):
    combobox = ttk.Combobox(window, values=[task for task in dict_of_tasks.values()])
    combobox.grid(row=row, column=column, padx=x, pady=y)
    combobox.bind("<<ComboboxSelected>>")
    return combobox

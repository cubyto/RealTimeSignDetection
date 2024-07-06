from PIL import Image, ImageTk


def read_image_and_resize(path, size):
    return ImageTk.PhotoImage(Image.open(path).resize(size, Image.ADAPTIVE))


def read_image(img):
    return ImageTk.PhotoImage(Image.open(img))


def center_window(window, app_width, app_height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = int((screen_width / 2) - (app_width / 2))
    y = int((screen_height / 2) - (app_height / 2))
    return window.geometry(f"{app_width}x{app_height}+{x}+{y}")

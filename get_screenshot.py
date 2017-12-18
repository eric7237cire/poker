import os

if os.name == 'nt':
    from PIL import ImageGrab
    import win32gui

toplist, winlist = [], []


def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))


def capture_screenshot(windows_title, output_file_path):
    win32gui.EnumWindows(enum_cb, toplist)

    # print(winlist)

    matching_hwnd_list = [(hwnd, title) for hwnd, title in winlist if windows_title.lower() in title.lower()]
    # just grab the hwnd for first window matching matching_hwnd_list
    matching_hwnd_list = matching_hwnd_list[0]
    hwnd = matching_hwnd_list[0]

    win32gui.SetForegroundWindow(hwnd)
    bbox = win32gui.GetWindowRect(hwnd)

    # print(bbox)
    img = ImageGrab.grab(bbox)
    # img.show()

    img.save(output_file_path)

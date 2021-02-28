# First import the library
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import shutil

import pyttsx3

from threading import Thread
import tkinter

record_thread = None

is_capture = None
start = False

engine = None

root = None

rgb_saving_frames = []
depth_saving_frames = []

# -------begin capturing and saving video
def startrecording(filename):
    global start
    start = True
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 6)

    # Start streaming
    pipeline.start(config)

    global rgb_saving_frames
    global depth_saving_frames

    try:
        while start:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            global is_capture
            # print(is_capture)
            if is_capture:
                print(is_capture)
                # rgb_saving_frames = np.append(rgb_saving_frames, color_image)
                depth_image = np.expand_dims(depth_image, axis=2)
                print(color_image.shape, depth_image.shape)
                # combined_depth_rgb_image = np.append(
                #     color_image, depth_image, axis=2
                # )
                rgb_saving_frames.append(color_image.copy())
                depth_saving_frames.append(depth_image.copy())
                print(len(rgb_saving_frames), len(depth_saving_frames))
                play_sound(len(rgb_saving_frames))
                is_capture = False

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET
            )

            # rgb_saving_frames.append(color_image)
            # Stack both images horizontally
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", images)
            cv2.waitKey(1)

        # depth_frames.append([color_image, depth_image])

    finally:
        pipeline.stop()
        np.savez(filename, rgb=rgb_saving_frames, depth=depth_saving_frames)
    start = False


def play_sound(text):
    global engine
    engine.say(str(text))
    engine.runAndWait()


def start_recording_proc(filename):
    if os.path.exists(filename):
        raise OSError(f"File path {filename} is existed.")

    global record_thread
    record_thread = Thread(target=startrecording, args=(filename,))
    record_thread.start()

    # p.start()


# -------end video capture and stop tk
def stoprecording():
    global start
    start = False

    global record_thread
    if record_thread is not None:
        record_thread.join()
        record_thread = None

    # e.set()

    global root
    root.quit()
    root.destroy()


def capture():
    global is_capture
    is_capture = True
    print("CAPTURE", is_capture)


def keypress_capture(e):
    capture()


def revert():
    global rgb_saving_frames
    global depth_saving_frames

    if len(rgb_saving_frames) > 0 and len(depth_saving_frames) > 0:
        rgb_saving_frames.pop()
        depth_saving_frames.pop()

    print(len(rgb_saving_frames), len(depth_saving_frames))
    play_sound(len(rgb_saving_frames))


def keypress_revert(e):
    revert()


def main():
    SAVING_DIR = "/home/disorn/code_save/python/realsense_cap/test-dataset-insp"

    total, used, free = shutil.disk_usage(SAVING_DIR)
    free_space_gib = free // (2 ** 30)
    if free_space_gib < 1:
        raise OSError("Dangerous: disk may have not enough spaces")

    SAVING_FILENAME = f"{SAVING_DIR}/2020-01-17-cucumber_soup-15"

    global root
    root = tkinter.Tk()
    root.geometry("%dx%d+0+0" % (100, 100))
    startbutton = tkinter.Button(
        root,
        width=10,
        height=1,
        text="START",
        command=lambda: start_recording_proc(SAVING_FILENAME),
    )
    stopbutton = tkinter.Button(
        root, width=10, height=1, text="STOP", command=stoprecording
    )
    capturebutton = tkinter.Button(
        root, width=10, height=1, text="CAPTURE", command=capture
    )
    startbutton.pack()
    stopbutton.pack()
    capturebutton.pack()

    global engine
    engine = pyttsx3.init()
    # engine.setProperty("volume", 0.9)  # Volume 0-1

    root.bind("<Return>", keypress_capture)
    root.bind("<BackSpace>", keypress_revert)

    # -------begin
    root.mainloop()


if __name__ == "__main__":
    main()
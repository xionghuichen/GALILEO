# Created by xionghuichen at 2023/8/5
# Email: chenxh@lamda.nju.edu.cn

import cv2
import os
def save_to_video(img_path, video_path, fps=30):
    """
    Save images to video.
    :param img_path: path to images
    :param video_path: path to video
    :param fps: frames per second
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4')
    video = cv2.VideoWriter(video_path, fourcc, fps, (500, 500))
    for idx in range(200):
        pathpath = os.path.join(img_path, f"{idx}.png")
        img = cv2.imread(pathpath)
        video.write(img)
    video.release()

save_to_video("./sl/expert-2", "./sl-tanh-expert.mp4", 10)

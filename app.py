from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from centerface import CenterFace
from typing import Dict, Tuple
import skimage.draw
import numpy as np
import cv2
import imageio.v2 as iio
from io import BytesIO

def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(
    frame,
    score,
    det_idx,
    x1,
    y1,
    x2,
    y2,
    replacewith: str = "blur",
    ellipse: bool = True,
    draw_scores: bool = False,
    ovcolor: Tuple[int] = (0, 0, 0),
    replaceimg=None,
    mosaicsize: int = 20,
):
    if replacewith == "solid":
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == "blur":
        bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
        blurred_box = cv2.blur(
            frame[y1:y2, x1:x2], (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
        )
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            # Get y and x coordinate lists of the "bounding ellipse"
            ey, ex = skimage.draw.ellipse(
                (y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2
            )
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == "img":
        target_size = (x2 - x1, y2 - y1)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:  # RGB
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:  # RGBA
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (
                1 - resized_replaceimg[:, :, 3:] / 255
            ) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
    elif replacewith == "mosaic":
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)
    elif replacewith == "none":
        pass
    if draw_scores:
        cv2.putText(
            frame,
            f"{score:.2f}",
            (x1 + 0, y1 - 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 255, 0),
        )


def anonymize_frame(
    dets, frame, mask_scale, replacewith, ellipse, draw_scores, replaceimg, mosaicsize
):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        # Clip bb coordinates to valid frame region
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
        draw_det(
            frame,
            score,
            i,
            x1,
            y1,
            x2,
            y2,
            replacewith=replacewith,
            ellipse=ellipse,
            draw_scores=draw_scores,
            replaceimg=replaceimg,
            mosaicsize=mosaicsize,
        )


centerface = CenterFace(backend="opencv")

app = FastAPI()

@app.post("/upload")
async def create_upload_file(file: UploadFile):
    frame = iio.imread(file.file)
    dets, _ = centerface(frame, threshold=0.2)
    anonymize_frame(
        dets,
        frame,
        mask_scale=1.3,
        replacewith="mosaic",
        ellipse=True,
        draw_scores=False,
        replaceimg=None,
        mosaicsize=15,
    )
    
    with BytesIO() as buf:
      iio.imwrite(buf, frame, plugin="pillow", format="JPEG")
      im_bytes = buf.getvalue()
    return Response(im_bytes, media_type="image/jpeg")
  
  
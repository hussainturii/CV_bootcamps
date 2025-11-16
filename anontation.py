import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

INPUT_PATH = "ye.jpeg"
OUTPUT_PATH = "annotated_numbered.jpg"

DP = 1.2
MIN_DIST = 14
PARAM1 = 50
PARAM2 = 28
MIN_RADIUS = 6
MAX_RADIUS = 22

FONT_SIZE = 18
TEXT_COLOR = (255, 255, 255)
OUTLINE_COLOR = (0, 0, 0)
OUTLINE_WIDTH = 2


def detect_circles(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=DP,
        minDist=MIN_DIST,
        param1=PARAM1,
        param2=PARAM2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    return circles.tolist()


def group_rows(circles, tolerance=20):
    if not circles:
        return []

    circles_sorted = sorted(circles, key=lambda c: c[1])

    rows = []
    current_row = [circles_sorted[0]]

    for c in circles_sorted[1:]:
        if abs(c[1] - current_row[-1][1]) <= tolerance:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]

    rows.append(current_row)

    final_rows = [sorted(r, key=lambda c: c[0]) for r in rows]

    return final_rows


def draw_numbers(pil_img, rows):
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    index = 1

    for row in rows:
        for (x, y, r) in row:
            s = str(index)

            bbox = draw.textbbox((0, 0), s, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            tx = x - w // 2
            ty = y - h // 2

            for ox in range(-OUTLINE_WIDTH, OUTLINE_WIDTH + 1):
                for oy in range(-OUTLINE_WIDTH, OUTLINE_WIDTH + 1):
                    if ox != 0 or oy != 0:
                        draw.text((tx + ox, ty + oy), s, font=font, fill=OUTLINE_COLOR)

            draw.text((tx, ty), s, font=font, fill=TEXT_COLOR)

            index += 1

    return pil_img


def main():
    if not os.path.exists(INPUT_PATH):
        print("Image not found:", INPUT_PATH)
        return

    pil = Image.open(INPUT_PATH).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    circles = detect_circles(bgr)
    print("Circles detected:", len(circles))

    rows = group_rows(circles, tolerance=20)
    total = sum(len(r) for r in rows)

    print("Rows detected:", len(rows))
    print("Final total:", total)

    annotated = draw_numbers(pil, rows)
    annotated.save(OUTPUT_PATH)

    print("Saved annotated image →", OUTPUT_PATH)


if __name__ == "__main__":
    main()

from PIL import Image


def calculate_crop_bbox(dog_bbox: tuple, im_size: tuple) -> tuple:
    """
    Try to get a rectangular shape out of the image containing only the dog centered

    :param dog_bbox: xmin, ymin, xmax, ymax of box body
    :param im_size: width, height of image
    :return: xmin, ymin, xmax, ymax to crop
    """
    xmin, ymin, xmax, ymax = dog_bbox
    width = xmax - xmin
    height = ymax - ymin
    if width == height:
        return dog_bbox
    elif width < height:
        diff = int((height - width) / 2)
        new_xmin = xmin - diff
        new_ymin = ymin
        new_xmax = xmax + diff
        new_ymax = ymax

        if new_xmin < 0:
            new_xmax = new_xmax - new_xmin
            new_xmin = 0
        elif new_xmax > im_size[0]:
            new_xmin = new_xmin - (new_xmax - im_size[0])
            new_xmax = im_size[0]

        new_xmin = max(0, new_xmin)
        new_xmax = min(im_size[0], new_xmax)
    elif width > height:
        diff = int((width - height) / 2)
        new_xmin = xmin
        new_ymin = ymin - diff
        new_xmax = xmax
        new_ymax = ymax + diff

        if new_ymin < 0:
            new_ymax = new_ymax - new_ymin
            new_ymin = 0
        elif new_ymax > im_size[1]:
            new_ymin = new_ymin - (new_ymax - im_size[1])
            new_ymax = im_size[1]

        new_ymin = max(0, new_ymin)
        new_ymax = min(im_size[1], new_ymax)

    return new_xmin, new_ymin, new_xmax, new_ymax


def crop(img_file, bbox_detected):
    im = Image.open(img_file)
    bbox = calculate_crop_bbox((bbox_detected.xmin, bbox_detected.ymin, bbox_detected.xmax, bbox_detected.ymax), im.size)
    im_cropped = im.crop(bbox)

    if im_cropped.mode in ("RGBA", "P"):
        im_cropped = im_cropped.convert("RGB")

    return im_cropped

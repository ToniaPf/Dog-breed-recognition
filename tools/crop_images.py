import os
from bs4 import BeautifulSoup
from PIL import Image


def crop_imgs_from_folder(in_folder, annot_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    for folder in os.listdir(annot_folder):
        os.makedirs(os.path.join(out_folder, folder))
        print(folder)

        for x in os.listdir(os.path.join(annot_folder, folder)):

            xml = os.path.join(annot_folder, folder, x)
            with open(xml) as f:
                s = BeautifulSoup(f)
                xmin = int(s.annotation.object.bodybndbox.xmin.text)
                xmax = int(s.annotation.object.bodybndbox.xmax.text)
                ymin = int(s.annotation.object.bodybndbox.ymin.text)
                ymax = int(s.annotation.object.bodybndbox.ymax.text)

            img_file = os.path.join(in_folder, folder, x.replace('.xml', ''))

            im = Image.open(img_file)
            im_cropped = im.crop((xmin, ymin, xmax, ymax))

            if im_cropped.mode in ("RGBA", "P"):
                im_cropped = im_cropped.convert("RGB")

            im_cropped.save(os.path.join(out_folder, folder, x.replace('.xml', '')))

    return


if __name__ == "__main__":
    crop_imgs_from_folder('../data/high/high-resolution',
                          '../data/high/high-annotations',
                          '../data/high/high-resolution-cropped')

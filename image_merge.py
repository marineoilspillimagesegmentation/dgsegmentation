import PIL.Image as Image
import os

IMAGES_PATH = '/scratch/oilspill/fc160/myproject/marine_oilspill_segmentation/DGNet_MarineOilSeg/ORIGINAL/'  
IMAGES_FORMAT = ['.png', '.bmp']  
IMAGE_WIDTH = 256  # width of image
IMAGE_HEIGHT = 256  # height of image
IMAGE_ROW = 1  # row of merged images
IMAGE_COLUMN = 2  # cloumn of merged images
IMAGE_SAVE_PATH = 'merge.jpg'  # path of merged images


image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

print(len(image_names))


#if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    #raise ValueError("incorrect number of image merge！")



def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_WIDTH, IMAGE_ROW * IMAGE_HEIGHT))  
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_WIDTH, (y - 1) * IMAGE_HEIGHT))
    return to_image.save(IMAGE_SAVE_PATH)  # save merged images


image_compose()  # function loaded

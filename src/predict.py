import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN
from util import getRandomStr
import os

IMAGE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/images"

def predict(images, model_name='psnr-small'):
    # initialize data generator
    img = Image.open(images)
    img = img.convert('RGB')
    lr_img = np.array(img)
    if (model_name == 'gans'):
      rdn = RRDN(weights=model_name)
    else:
      rdn = RDN(weights=model_name)
    sr_img = rdn.predict(lr_img)
    output = Image.fromarray(sr_img)
    output_file_path = IMAGE_PATH + getRandomStr(15) + '.jpg'
    output.save(output_file_path)
    
    return output_file_path
    

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from rembg.bg import remove
import numpy as np
import io
from PIL import Image
from multipledispatch import dispatch
"""
[summary]
    import ImageUtil as iu
    
    input_path = 'C:\\Users\\hyunjong\\Desktop\\temp\\mask2.jpg'
    output_path = 'C:\\Users\\hyunjong\\Desktop\\temp\\mask2_out.png'

    #Use path
    iu.EraseBackground.save_image(input_path, output_path)
    
    #Use np image file for input
    input_image = np.fromfile(input_path)
    iu.EraseBackground.save_image(input_image, output_path)

    #Get background erased np image
    input_image = np.fromfile(input_path)
    background_erased_image = iu.get_image(input_image)

    #pip install python-dispatch
"""
class EraseBackground():
    DEFAULT_MODEL_NAME = 'u2net'
    HUMAN_MODEL_NAME = 'u2net_human_seg'

    @classmethod
    #@dispatch(str, str)
    def save_image(cls, input_image_path : str, output_image_path : str):
        input_image = np.fromfile(input_image_path)
        erase_result = remove(input_image, model_name=cls.HUMAN_MODEL_NAME)
        background_erased_image = Image.open(io.BytesIO(erase_result)).convert("RGBA")
        background_erased_image.save(output_image_path)
    
    # #@classmethod
    # @dispatch(np.ndarray, str)
    # def save_image(cls, input_image : np.ndarray, output_image_path : str):
    #     erase_result = remove(input_image, model_name=cls.HUMAN_MODEL_NAME)
    #     background_erased_image = Image.open(io.BytesIO(erase_result)).convert("RGBA")
    #     background_erased_image.save(output_image_path)

    @classmethod
    #@dispatch(np.ndarray)
    def get_image(cls, input_image : np.ndarray) -> np.ndarray:
        erase_result = remove(input_image, model_name=cls.HUMAN_MODEL_NAME)
        background_erased_image = Image.open(io.BytesIO(erase_result)).convert("RGBA")
        return background_erased_image

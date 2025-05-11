import torchvision.models as models
from preprocessing_from_db import custom_preprocessing
def cnn_from_camera_stream(image):
    proccess_image=custom_preprocessing(image)
    model = models.shufflenet_v2_x1_0(pretrained=False)
    output=model(proccess_image)
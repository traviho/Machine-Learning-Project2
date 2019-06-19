from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
import pickle

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

def loadPixelClass():
    with open("classArray", 'rb') as fp:
        classes = pickle.load(fp)
    return classes

def fastApproach(x, c):
    nr = 0
    #print(c)
    for i in range(len(c)):
        if np.array_equal(x, c[i]):
            nr = i
            break
    array = [0]*21
    array[nr] = 1
    return array

def ld(path):
    path_to_y_train = "./VOCdevkit/VOC2012/SegmentationClass"
    path_to_x_train = "./VOCdevkit/VOC2012/JPEGImages"
    xtmp = []
    ytmp = []
    with open(path) as file:
        for line in file:
            x = line.strip()
            ytmp.append(path_to_y_train+"/{}.png".format(x))
            xtmp.append(path_to_x_train+"/{}.jpg".format(x))
    return xtmp, ytmp

def load_data():
    path = "./VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
    val_path = "./VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

    x_train = []
    y_train = []    #(1464, 224, 320)
    x_val = []
    y_val = []
    dim = (320, 224)
    xPath, yPath = ld(path)
    xValPath, yValPath = ld(val_path)
    c = loadPixelClass()
    openImg = lambda path: Image.open(path)
    openAndResize = lambda path: openImg(path).resize(dim)
    getRGB = lambda path: np.array(openAndResize(path).convert("RGB"))
    formatClassLabel = lambda img: [[fastApproach(img[i][j],c) for j in range(dim[0])] for i in range(dim[1])]
    mapPathFormatClassLbl = lambda path : np.array(list(map(formatClassLabel, list(map(getRGB, path))))) 
    """
        yImage = getRGB(yPath)
        picClass = [[fastApproach(yImage[i][j],c) for j in range(dim[0])] for i in range(dim[1])]
        yImage = np.array(picClass)
        print(yImage.shape)
        y_train.append(yImage)
    """
    #Batch-Size
    print("dataset size xtrain: {}\t xVal: {}".format(len(xPath), len(xValPath)))
    xPath = xPath[:100]
    yPath = xPath[:100]
    xValPath = xPath[:100]
    yValPath = xPath[:100]
    

    print("path-labels: {}".format(len(xPath)))
    #l = []
    #for p in yPath:
    #    print(p)
        #l.append(getRGB(p))

    #yImgs = list(map(getRGB, yPath))
    #np.array(list(map(formatClassLabel, yImgs)))

    #yImgs = list(map(getRGB, yValPath))
    #y_val = np.array(list(map(formatClassLabel, yImgs)))
    y_train = mapPathFormatClassLbl(yPath)
    print("finished y training data")
    y_val = mapPathFormatClassLbl(yValPath)
    print("finished y val data")
    x_train = np.array(list(map(lambda p: np.array(openAndResize(p)), xPath)))
    x_val = np.array(list(map(lambda p: np.array(openAndResize(p)), xValPath)))
    """
        yValImg = getRGB(yValPath)
        picClass = [[fastApproach(tmp[i][j],c) for j in range(dim[0])] for i in range(dim[1])]
        yValImg = np.array(picClass)
        y_val.append(yValImg)
        print(yValImg.shape)
    """


    print(y_train[0][0][0])
    y_val = np.array(y_val)
    allData = [x_train, y_train, x_val, y_val]
    with open("preprocessedImages", 'wb') as fp:
        pickle.dump(allData, fp)
    return x_train, y_train, x_val, y_val


def load():
    try:
        with open("preprocessedImages", 'rb') as fp:
            classes = pickle.load(fp)
            return classes[0], classes[1], classes[2], classes[3] 
    except:
        return load_data()
# load your data
x_train, y_train, x_val, y_val = load()


#print(x_train)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
#print(y_val.shape)
#x_train = x_train.reshape(x_train.shape[0], 320, 240, 3)
#x_val = x_val.reshape(x_val.shape[0], 320, 240, 3)
x_train = x_train.astype('float32')
x_test = x_val.astype('float32')
x_train /= 255
x_test /= 255


# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# define model
model = Unet(BACKBONE, classes=21 )
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

# fit model
model.fit(
    x=x_train,
    y=y_train,
    batch_size=1,
    epochs=100,
    validation_data=(x_val, y_val),
)
model.save_weights("savedModel")
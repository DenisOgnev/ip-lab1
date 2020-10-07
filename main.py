import cv2
import random as r
import math
import numpy as np
import time

def addSaltAndPepper(image, n):
    heigth = image.shape[0]
    width = image.shape[1]
    numberOfNoisyPixels = r.randrange(0, math.floor(heigth * width / 100)) * n
    counterOfAddedNoisyPixels = 0

    returnImage = np.array(image)

    for i in range(heigth * width):
        if counterOfAddedNoisyPixels < numberOfNoisyPixels:
            if r.random() > 0.7:
                returnImage[r.randrange(0, heigth)][r.randrange(0, width)] = 255
            else:
                returnImage[r.randrange(0, heigth)][r.randrange(0, width)] = 0
            counterOfAddedNoisyPixels += 1
        else:
            break
    
    return returnImage

def immse(image1, image2):
    mse = 0.0
    if np.shape(image1)[0] != np.shape(image2)[0] or np.shape(image1)[1] != np.shape(image1)[1]:
        raise Exception("Wrong dismensions")
    for i in range(np.shape(image1)[0]):
        for j in range(np.shape(image1)[1]):
            mse = mse + ((sum(image1[i][j]) / 3 - sum(image2[i][j]) / 3) ** 2)

    mse = mse / (np.shape(image1)[0] * np.shape(image2)[1])

    return mse

def immseHSV(image1, image2):
    mse = 0.0
    if np.shape(image1)[0] != np.shape(image2)[0] or np.shape(image1)[1] != np.shape(image1)[1]:
        raise Exception("Wrong dismensions")
    for i in range(np.shape(image1)[0]):
        for j in range(np.shape(image1)[1]):
            mse = mse + ((int(image1[i][j][2]) - int(image2[i][j][2])) ** 2)

    mse = mse / (np.shape(image1)[0] * np.shape(image2)[1])

    return mse

def shadesOfGray(image): #average (R + G + B) / 3
    returnImage = np.array(image)

    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            returnImage[i][j] = [np.sum(image[i][j]) / 3, np.sum(image[i][j]) / 3, np.sum(image[i][j]) / 3]

    return returnImage

def rbg2hsv(image):
    returnImage = np.array(image)
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            hsvR = image[i][j][2] / 255
            hsvG = image[i][j][1] / 255
            hsvB = image[i][j][0] / 255
            hsvCmax = max(hsvR, hsvG, hsvB) 
            hsvCmin = min(hsvR, hsvG, hsvB)
            hsvDelta = hsvCmax - hsvCmin
            if hsvDelta == 0:
                hsvH = 0
            elif hsvCmax == hsvR:
                hsvH = 60 * (((hsvG - hsvB) / hsvDelta) % 6)
            elif hsvCmax == hsvG:
                hsvH = 60 * (((hsvB - hsvR) / hsvDelta) + 2)
            elif hsvCmax == hsvB:
                hsvH = 60 * (((hsvR - hsvG) / hsvDelta) + 4)
            if hsvCmax == 0:
                hsvS = 0
            else:
                hsvS = hsvDelta / hsvCmax
            hsvV = hsvCmax
            returnImage[i][j] = [hsvH, hsvS, hsvV]
    return returnImage

def brightnessIncreaseRGB(image, numberOfIncrease):
    returnImage = np.array(image)
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            if returnImage[i][j][0] + numberOfIncrease > 255:
                returnImage[i][j][0] = 255
            elif returnImage[i][j][0] + numberOfIncrease < 0:
                returnImage[i][j][0] = 0
            else:
                returnImage[i][j][0] = returnImage[i][j][0] + numberOfIncrease

            if returnImage[i][j][1] + numberOfIncrease > 255:
                returnImage[i][j][1] = 255
            elif returnImage[i][j][1] + numberOfIncrease < 0:
                returnImage[i][j][1] = 0
            else:
                returnImage[i][j][1] = returnImage[i][j][1] + numberOfIncrease

            if returnImage[i][j][2] + numberOfIncrease > 255:
                returnImage[i][j][2] = 255
            elif returnImage[i][j][2] + numberOfIncrease < 0:
                returnImage[i][j][2] = 0
            else:
                returnImage[i][j][2] = returnImage[i][j][2] + numberOfIncrease
            
    return returnImage

def brightnessIncreaseHSV(image, numberOfIncrease):
    returnImage = np.array(image)
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            if returnImage[i][j][2] + numberOfIncrease > 100:
                returnImage[i][j][2] = 100
            elif returnImage[i][j][2] + numberOfIncrease < 0:
                returnImage[i][j][2] = 0
            else:
                returnImage[i][j][2] = returnImage[i][j][2] + numberOfIncrease
    
    return returnImage


def main():

    firstImageForMseTest = cv2.imread("1.png")
    secondImageForMseTest = cv2.imread("2.png")
    imageForShadeOfGrayFilter = cv2.imread("ex.jpg")
    imageForConverting = cv2.imread("ex.jpg")


    handle = open("result.txt", "w")

    handle.write("Mse test: " + str(immse(firstImageForMseTest, secondImageForMseTest))) #MSE TEST


    startTime = time.time() #SHADES OF GRAY
    myImageOfShadeOfGrayFilter = shadesOfGray(imageForShadeOfGrayFilter)
    endTime = time.time() - startTime
    cv2.imwrite("MyShadesOfGray.jpg", myImageOfShadeOfGrayFilter)
    handle.write("\n\nИзображение в оттенках серого через мой фильтр сохранено в файл MyShadesOfGray.jpg, время выполнения: " + str(endTime) + " секунд")
    startTime = time.time()
    openCvImageOfShadeOfGrayFilter = cv2.cvtColor(imageForShadeOfGrayFilter, cv2.COLOR_BGR2GRAY)
    endTime = time.time() - startTime
    cv2.imwrite("opencvShadesOfGray.jpg", openCvImageOfShadeOfGrayFilter)
    handle.write("\nИзображение в оттенках серого через фильтр opencv сохранено в файл opencvShadesOfGray.jpg, время выполнения: " + str(endTime) + " секунд")
    handle.write("\nРазница в цвете обусловлена алгоритмом преобразования, в моей реализации используется average (R + G + B) / 3")
    handle.write(", а разница в скорости из-за реализации, возможно реализация cv2 более быстрее проходит массивы")


    startTime = time.time() #BRG -> HSV
    hsvImage = rbg2hsv(imageForConverting)
    endTime = time.time() - startTime
    cv2.imwrite("MyHsvImage.jpg", hsvImage)
    handle.write("\n\nИзображение переведенное в HSV через мой фильтр сохранено в MyHsvImage, время перевода: " + str(endTime) + " секунд")
    startTime = time.time()
    opencvHsvImage = cv2.cvtColor(imageForConverting, cv2.COLOR_BGR2HSV)
    endTime = time.time() - startTime
    cv2.imwrite("opencvHsvImage.jpg", opencvHsvImage)
    handle.write("\nИзображение переведенное в HSV средствами cv2 сохранено в opencvHsvImage.jpg, время перевода: " + str(endTime) + " секунд")
    handle.write("\nРазница в цвете возможно из-за того, что opencv хранит RGB изображения как BGR. Проверял функцию собственную перевода попиксельно по калькуляторам, перевод верный")
    startTime = time.time()
    rgbIncreaseBrightness = brightnessIncreaseRGB(imageForConverting, 51)
    endTime = time.time() - startTime
    cv2.imwrite("rgbIncreaseBrightness.jpg", rgbIncreaseBrightness)
    handle.write("\nRGB изображение с увеличенной яркостью сохранено в файл rgbIncreaseBrightness.jpg, время выполнения: " + str(endTime) + " секунд")
    startTime = time.time()
    hsvIncreaseBrightness = brightnessIncreaseHSV(opencvHsvImage, 20)
    endTime = time.time() - startTime
    cv2.imwrite("hsvIncreaseBrightness.jpg", hsvIncreaseBrightness)
    handle.write("\nRGB изображение с увеличенной яркостью сохранено в файл hsvIncreaseBrightness.jpg, время выполнения: " + str(endTime) + " секунд")
    handle.write("\nMse RGB изображение с увеличенной яркостью и неувеличенной: " + str(immse(imageForConverting, rgbIncreaseBrightness)))
    handle.write("\nMse HSV изображение с увеличенной яркостью и неувеличенной: " + str(immse(opencvHsvImage, hsvIncreaseBrightness)))
    handle.write("\nMse RGB и HSV изображений с увеличенной яркостью: " + str(immse(rgbIncreaseBrightness, hsvIncreaseBrightness)))
    handle.write("\nРазница в том, что увеличение якрости RGB увеличивает все три компоненты R, G и B, а увеличение яркости HSV увеличивает только значение V - яркости")


    handle.close()
    

    #cv2.imshow("image", image1)

    #while True:
        #k = cv2.waitKey(0)
        #if k == 27:
            #cv2.destroyAllWindows()
            #break
        #elif k == 115:
            #cv2.imwrite("ex1.jpg", image1)
            #cv2.destroyAllWindows()
            #break

    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
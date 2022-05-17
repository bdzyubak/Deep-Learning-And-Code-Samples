import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
import scipy.ndimage
import scipy.misc

import math
import scipy.ndimage.morphology
import skimage
import cv2


def plsdraw(Image, mask=0, referenceMask=0):
    if np.ndim(Image) == 3:
        ind = np.argmin(Image.shape)
        zsze = Image.shape[ind]
        if ind == 0:
            Image = Image.transpose(1, 2, 0)
            if not isinstance(mask, int):
                mask = mask.transpose(1, 2, 0)
            if not isinstance(referenceMask, int):
                referenceMask = referenceMask.transpose(1, 2, 0)
    else:
        zsze = 1

    for l in range(zsze):
        if zsze == 1:
            I = Image
        else:
            I = Image[:,:,l]

        plt.figure(dpi=150)

        if np.max(I) > 1:
            plt.imshow(I, vmin=np.percentile(I, 5), vmax=np.percentile(I, 95), cmap='gray')
        else:
            plt.imshow(I, vmin=0, vmax=1, cmap='gray')
        plt.axis('off')

        if not isinstance(mask, int):
            if zsze == 1:
                M = np.squeeze(mask)
            else:
                M = np.squeeze(mask[:,:,l])

            plt.contour(M>0.5,levels=[0],colors='b')

        if not isinstance(referenceMask, int):
            if zsze == 1:
                Ref = np.squeeze(referenceMask)
            else:
                Ref = np.squeeze(referenceMask[:,:,l])

            plt.contour(Ref>0.5, levels=[0], colors='g')

        plt.show()

def closeall():

    plt.close('all')

    return

def rescale_values(Mag,Elast,Conf,ROI=None):
    Mag = Mag / np.percentile(Mag, 98)
    Conf = Conf / 1000
    Elast = Elast / 8000

    if ROI is not None:
        ROI = (1 * (ROI > 0)).astype(dtype="float32")
        ROI[Conf < 0.9] = 0
    return Mag, Elast, Conf, ROI


def rescale_values_fw(water,ffrac,ROI=None,r2star=None):
    water = water / np.percentile(water, 98)
    ffrac = ffrac / np.percentile(ffrac, 98)

    if r2star is not None:
        r2star = r2star / np.percentile(r2star, 98)

    if ROI is not None:
        ROI = (1 * (ROI > 0)).astype(dtype="float32")

    return water, ffrac, r2star, ROI


def cleanMask(mask):
    if np.ndim(mask)==3:
        zsze = mask.shape[2]
    else:
        zsze = 1

    cleanedMask = np.zeros_like(mask)

    for l in range(zsze):
        cleaned = mask[:, :, l]
        cleaned = scipy.ndimage.binary_closing(cleaned, structure=np.ones((5, 5)))
        cleaned = np.invert(morphology.remove_small_objects(np.invert(cleaned), min_size=150, connectivity=2))  # remove small objects
        cleaned = scipy.ndimage.binary_opening(cleaned, structure=np.ones((5, 5)))
        cleanedMask[:,:,l] = morphology.remove_small_objects(cleaned, min_size=150, connectivity=2)  # remove small objects

    return cleanedMask


def removeAndClean(Mag,Elast,Mask):
    for id in range(Mag.shape[0]):
        if np.sum(Mask[id,:,:,:]) > 0:
            M = Mag[id,:,:,:]
            E = Elast[id,:,:,:]
            mm = Mask[id,:,:,:]
            MDiff = np.abs(M - np.mean(M[mm]))
            EDiff = np.abs(E - np.mean(E[mm]))
            Outlier = np.logical_or(MDiff>np.percentile(MDiff[mm],95), EDiff>np.percentile(EDiff[mm],95))
            SharedOutlier = np.logical_and(MDiff>np.percentile(MDiff[mm],80), EDiff>np.percentile(EDiff[mm],80))
            AllOutliers = np.logical_or(Outlier,SharedOutlier)
            newmask = np.logical_and(mm,np.invert(AllOutliers))
            Mask[id,:,:,:] = newmask

    return Mask


def saveMaskContour(mask,filename='Image1.png'):
    fig = plt.figure(dpi=150,frameon=False)
    plt.axis('off')
    plt.contour(mask, levels=[0], colors='b')
    plt.gca().invert_yaxis() # because it's a plot not image, python reverses it upside down. Curiously, this doesn't happen with plots overlaid on images
    fig.savefig(filename, transparent=True)


def saveMaskedImage(I,mask,referenceMask=0,filename='MaskedImage1.png'):
    fig = plt.figure(dpi=150,frameon=False)
    plt.imshow(I, vmin=np.percentile(I, 5), vmax=np.percentile(I, 95), cmap='gray')
    plt.axis('off')

    if not isinstance(mask, int):
        plt.contour(mask,levels=[0],colors='b')

    if not isinstance(referenceMask, int):
        plt.contour(referenceMask, levels=[0], colors='g')

    fig.savefig(filename, transparent=True)


def reshapeAsSlices(Itemp):
    NumSlices = 4
    NumExams = round(Itemp.shape[0]/NumSlices)

    I = np.zeros((NumExams,Itemp.shape[1],Itemp.shape[2],NumSlices),dtype=type(Itemp[0,0,0]))

    for e in range(NumExams):
        for l in range(NumSlices):
            I[e,:,:,l] = Itemp[e*NumSlices + l, :, :]
    return I


def reshapeAsImages(Itemp):
    NumSlices = 4
    NumExams = round(Itemp.shape[0])
    I = np.zeros((NumExams*NumSlices, Itemp.shape[1], Itemp.shape[2]), dtype=type(Itemp[0, 0, 0]))

    for e in range(NumExams):
        for l in range(NumSlices):
            I[e * NumSlices + l, :, :] = Itemp[e, :, :, l]

    return I


def shiftImage(I,xshift,yshift):
    shift = yshift
    if yshift > 0:
        Magtemp = np.pad(I[yshift:, :], [(0, np.abs(yshift)), (0, 0)], mode="constant")
    elif yshift < 0:
        Magtemp = np.pad(I[:(I.shape[0] - np.abs(yshift)), :], [(np.abs(yshift), 0), (0, 0)], mode="constant")
    else:
        Magtemp = I

    if xshift > 0:
        Magtemp = np.pad(Magtemp[:, xshift:], [(0, 0), (0, np.abs(xshift))], mode="constant")
    elif xshift < 0:
        Magtemp = np.pad(Magtemp[:, :(I.shape[1] - np.abs(xshift))], [(0, 0), (np.abs(xshift), 0)], mode="constant")

    return Magtemp


def padImage(I,xpad,ypad):
    # Pad in x and y using forward-transform coordinates for compressing (unevenly) the not-truly-square body
    Ipad = np.pad(I, [(int(math.ceil(xpad/2)), int(math.floor(xpad/2))), (int(math.ceil(ypad/2)), int(math.floor(ypad/2)))], mode="constant")

    return Ipad


def cropImage(I,xmin,xmax,ymin,ymax):
    # Crop non-square body in x and y then pad to square for stretching
    xscale = I.shape[1] / (xmax - xmin)
    yscale = I.shape[0] / (ymax - ymin)

    Icrop = I[ymin:ymax, xmin:xmax]

    if xscale < yscale:
        padSize = (Icrop.shape[1] - Icrop.shape[0]) / 2
        Icrop = np.pad(Icrop, [(math.floor(padSize), math.ceil(padSize)), (0, 0)], mode="constant")
    else:
        padSize = (Icrop.shape[0] - Icrop.shape[1]) / 2
        Icrop = np.pad(Icrop, [(0, 0), (math.floor(padSize), math.ceil(padSize))], mode="constant")

    return Icrop


def elscale(el,xsze):
    scaling = xsze/256
    scaled = np.round(el*scaling)
    return scaled


def remove_separate_ROIs(imask,xsze,roi_fraction=1):
    imask = skimage.morphology.binary_erosion(imask, skimage.morphology.disk(elscale(20, xsze)))
    if np.sum(imask) > 0:
        imask_labeled = skimage.measure.label(imask.astype(int))
        R = skimage.measure.regionprops_table(imask_labeled, properties=['area'])
        ind = [ind for ind, x in enumerate(R['area']) if x < roi_fraction*max(R['area'])] # Remove objects smaller than roi_fraction of biggest object (body)

        for i in ind:
            imask[imask_labeled == i + 1] = 0
        imask = skimage.morphology.binary_dilation(imask, skimage.morphology.disk(elscale(22, xsze)))

    return imask


def make_bodymask(M,data_type):
    xsze = M.shape[1]
    zsze = M.shape[0]
    body = np.zeros_like(M)
    bodymask = np.zeros_like(M)
    for l in range(zsze):
        I = M[l,:,:]*256
        element = skimage.morphology.disk(3)
        if data_type[l] == 'EPI' or data_type[l] == 'IQMRE2D':
            threshold = 15
        else:
            threshold = 10

        imask = skimage.morphology.binary_closing(I > threshold, element)
        imask = scipy.ndimage.binary_fill_holes(imask)

        imask = remove_separate_ROIs(imask,xsze,roi_fraction=1)
        body[l,:,:] = imask

    for l in range(int(zsze/4)):
        bodymask[l*4:(l*4+4),:,:] = np.transpose(np.dstack([np.mean(body[l*4:(l*4+4),:,:], axis=0)] * 4),(2,0,1))

    return bodymask


def cropZoomIn(I, bodymask, T2 = None):
    if T2 is None:
        Trans = np.zeros(6,dtype=int)

        xmin = min(np.where(bodymask)[1])
        xmax = max(np.where(bodymask)[1])
        ymin = min(np.where(bodymask)[0])
        ymax = max(np.where(bodymask)[0])

        xcenter = int(math.floor(xmin + (xmax - xmin) / 2))
        ycenter = int(math.floor(ymin + (ymax - ymin) / 2))

        xshift = int(xcenter - I.shape[1] / 2)  # shift left by this much
        yshift = int(ycenter - I.shape[0] / 2)  # shift up by this much

        Trans[0] = xshift
        Trans[1] = yshift

        I = shiftImage(I, xshift, yshift)

        xmin -= xshift
        xmax -= xshift
        ymin -= yshift
        ymax -= yshift

        Trans[2] = xmin
        Trans[3] = xmax
        Trans[4] = ymin
        Trans[5] = ymax

    else:
        Trans = T2
        xshift = Trans[0]
        yshift = Trans[1]

        I = shiftImage(I, xshift, yshift)

        xmin = Trans[2]
        xmax = Trans[3]
        ymin = Trans[4]
        ymax = Trans[5]

    MagTempPad = cropImage(I, xmin, xmax, ymin, ymax)

    MagTempFinal = cv2.resize(MagTempPad, (I.shape[0], I.shape[1]), interpolation=cv2.INTER_CUBIC)

    return MagTempFinal, Trans


def padZoomOut(I, Transform):
    xpad = np.abs(I.shape[1] - (Transform[3] - Transform[2])).astype(int)
    ypad = np.abs(I.shape[0] - (Transform[5] - Transform[4])).astype(int)

    if xpad>ypad:
        Pad = ypad
    else:
        Pad = xpad

    Iscale = cv2.resize(I.astype(float), (I.shape[1] - Pad, I.shape[0] - Pad), interpolation=cv2.INTER_CUBIC)

    Ipad = padImage(Iscale, Pad, Pad)

    xshift = Transform[0]
    yshift = Transform[1]

    Ishift = shiftImage(Ipad, xshift, yshift) # Already negated

    return Ishift

## Rescaling of all images to the same coordinate system and transforming back by reversing forward-transforms
def cropRescaleAll(MagIn,ElastIn,ROIIn,ConfIn,data_type):
    # Index through all images to do forward transform and store transform coordinates
    Mag = np.array(MagIn)
    Elast = np.array(ElastIn)
    ROI = np.array(ROIIn)
    Conf = np.array(ConfIn)

    MagOut = np.zeros_like(Mag)
    ElastOut = np.zeros_like(Elast)
    ROIOut = np.zeros_like(ROI)
    ConfOut = np.zeros_like(Conf)

    ImageTransforms = np.zeros((Mag.shape[0],6),dtype=int)
    bodymask = make_bodymask(Mag,data_type)
    for e in range(Mag.shape[0]):
        print(e)
        Magtemp, Transform = cropZoomIn(Mag[e, :, :], bodymask[e, :, :])
        Conftemp, _ = cropZoomIn(Conf[e, :, :], bodymask[e, :, :], Transform)
        Elasttemp, _ = cropZoomIn(Elast[e, :, :], bodymask[e, :, :], Transform)
        ROItemp, _ = cropZoomIn(ROI[e, :, :], bodymask[e, :, :], Transform)

        MagOut[e, :, :] = Magtemp
        ElastOut[e, :, :] = Elasttemp
        ROIOut[e, :, :] = ROItemp > 0.5
        ConfOut[e, :, :] = Conftemp

        ImageTransforms[e,:] = Transform

    return MagOut, ElastOut, ROIOut, ConfOut, ImageTransforms


def uncropUpscaleAll(ROIIn,ImageTransforms):
    # Index through all masks to do reverse transform
    ROI = np.array(ROIIn)

    for e in range(ROI.shape[0]):
        Transform = ImageTransforms[e,:]
        Transform[0:2] = -Transform[0:2].astype(int) # Shift back

        # for l in range(4):
        ROItemp = padZoomOut(ROI[e, :, :], Transform)

        ROI[e, :, :] = ROItemp > 0.5

    return ROI

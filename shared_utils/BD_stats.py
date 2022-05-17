import numpy as np

def calcStiffSlice(Elast,Mask):
    # Use function to calculate area-weighted stiffness and ROI area for one or more of Exams
    NumExams = int(Mask.shape[0])
    Stiffs = np.ones((NumExams))
    Areas = np.ones((NumExams))

    for i in range(NumExams):
            ElastSlice = Elast[i,:,:]

            MaskSlice = Mask[i,:,:]>0.5
            StiffInMask = ElastSlice[MaskSlice]
            if len(StiffInMask)>0:
                Stiffs[i] = StiffInMask.mean()*8 # Rescale to kPa
                Areas[i] = StiffInMask.shape[0]  # Get Mask Area in Pixels
            else:
                Stiffs[i] = 0
                Areas[i] = 0

    return Stiffs, Areas

def calcStiff(Elast,Mask):
    NumExams = Elast.shape[0]
    hasSlices = Elast.ndim == 4

    if hasSlices:
        NumSlices = Elast.shape[-1]  # -1: is same as -1 - last element in list but accounts for empty lists/strings
    else:
        NumSlices = 1

    # Use function to calculate area-weighted stiffness and ROI area for one or more of Exams
    Stiffnesses = np.zeros(NumExams)
    Areas = np.zeros(NumExams)
    StiffSlices = np.zeros((NumExams, NumSlices))
    AreaSlices = np.zeros((NumExams, NumSlices))

    for i in range(NumExams):
        if hasSlices:
            for l in range(4):
                ElastSlice = Elast[i,:,:,l]

                MaskSlice = Mask[i,:,:,l]>0.5
                StiffInMask = ElastSlice[MaskSlice]
                if len(StiffInMask)>0:
                    StiffSlices[i,l] = StiffInMask.mean()*8 # Rescale to kPa
                    AreaSlices[i, l] = StiffInMask.shape[0]  # Get Mask Area in Pixels
                else:
                    StiffSlices[i,l] = 0
                    AreaSlices[i,l] = 0
        elif ~hasSlices:
                ElastSlice = Elast[i,:,:]

                MaskSlice = Mask[i,:,:]>0.5
                StiffInMask = ElastSlice[MaskSlice]
                if len(StiffInMask)>0:
                    StiffSlices[i] = StiffInMask.mean()*8 # Rescale to kPa
                    AreaSlices[i] = StiffInMask.shape[0]  # Get Mask Area in Pixels
                else:
                    StiffSlices[i] = 0
                    AreaSlices[i] = 0

        Areas = np.sum(AreaSlices, axis=1)

        Areas[Areas == 0] = np.nan

        Stiffnesses = np.nansum(np.multiply(StiffSlices, AreaSlices), axis=1) / Areas  # autoyields nans for Area=0

    return Stiffnesses, Areas, StiffSlices, AreaSlices

def rdnum(number):
    # Return a number with as string with two decimal places for display
    return format(number, '0.2f')

def prcdiff(AutoVals, ManualVals):
    # Use this function to calculate an array of percent differences between two arrays of measurements
    Average = (AutoVals + ManualVals)/2
    if isinstance(Average, np.ndarray):
        Average[Average==0] = np.nan

    PercentDiffs = (AutoVals - ManualVals)/Average*100
    return PercentDiffs

def getDice(CandidateMask,ReferenceMask):
    ReferenceMask = np.squeeze(ReferenceMask).astype(dtype=float)
    CandidateMask = np.squeeze(CandidateMask).astype(dtype=float)

    NumExams = CandidateMask.shape[0]
    hasSlices = CandidateMask.ndim == 4

    if hasSlices:
        NumSlices = CandidateMask.shape[-1] # -1: is same as -1 - last element in list but accounts for empty lists/strings
    else:
        NumSlices = 1

    dice = np.zeros(NumExams)
    diceSlices = np.zeros((NumExams,NumSlices))

    for i in range(NumExams):
        if hasSlices:
            for l in range(NumSlices):
                if np.sum(CandidateMask[i, :, :, l] > 0.5) == 0 and np.sum(ReferenceMask[i, :, :, l] > 0.5) == 0:
                    diceSlices[i, l] = 1
                else:
                    union = np.sum(CandidateMask[i, :, :, l] > 0.5) + np.sum(ReferenceMask[i, :, :, l] > 0.5)
                    intersection = np.sum((CandidateMask[i, :, :, l] > 0.5) & (ReferenceMask[i, :, :, l] > 0.5))

                    if union == 0:
                        diceSlices[i,l] = 0
                    else:
                        diceSlices[i,l] = 2 * intersection / union

        elif ~hasSlices:
            if np.sum(CandidateMask[i, :, :] > 0.5) == 0 and np.sum(ReferenceMask[i, :, :] > 0.5) == 0:
                diceSlices[i] = 1
            else:
                union = np.sum(CandidateMask[i,:,:]>0.5) + np.sum(ReferenceMask[i,:,:]>0.5)
                intersection = np.sum((CandidateMask[i,:,:]>0.5) & (ReferenceMask[i,:,:]>0.5))

                if union == 0:
                    diceSlices[i] = 0
                else:
                    diceSlices[i] = 2*intersection/union

        dice[i] = np.mean(diceSlices[i])

    return np.mean(dice), dice, diceSlices


def getStiffDiff(Elast,CandidateMask,ReferenceMask):
    Elast = np.squeeze(Elast).astype(dtype=float)
    ReferenceMask = np.squeeze(ReferenceMask).astype(dtype=float)
    CandidateMask = np.squeeze(CandidateMask).astype(dtype=float)

    NumExams = CandidateMask.shape[0]
    stiff_diff_cases = np.zeros((NumExams))

    for i in range(NumExams):
        if np.sum(CandidateMask[i, :, :] > 0.5) == 0 and np.sum(ReferenceMask[i, :, :] > 0.5) == 0:
            # No ROI for both mask types
            stiff_diff_cases[i] = 0
        elif (np.sum(CandidateMask[i, :, :] > 0.5) == 0 and np.sum(ReferenceMask[i, :, :] > 0.5) > 125) or (np.sum(CandidateMask[i, :, :] > 0.5) >125 and np.sum(ReferenceMask[i, :, :] > 0.5) == 0):
            # Missing or extra ROI
            stiff_diff_cases[i] = float('NaN')
        else:
            elast_image = Elast[i, : :]
            stiff_diff_cases[i] = (elast_image[ReferenceMask[i, :, :]>0.5].mean() - elast_image[CandidateMask[i, :, :]>0.5].mean())/elast_image[ReferenceMask[i, :, :]>0.5].mean()*100

    stiff_diff = np.nanmean(stiff_diff_cases)
    stiff_diff_stdev = np.nanstd(stiff_diff_cases)

    return stiff_diff, stiff_diff_stdev, stiff_diff_cases
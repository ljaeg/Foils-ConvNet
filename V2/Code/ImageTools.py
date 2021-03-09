import numpy as np
from scipy.ndimage import measurements as meas
from scipy import ndimage as ndi

def GetRandomFOV(img, FOVSize):
    x = np.random.randint(0, img.shape[0])
    y = np.random.randint(0, img.shape[1])
    mmin, mmax, nmin, nmax = MakeBounds(x, y, img.shape, FOVSize)
    return img[mmin:mmax, nmin:nmax]

def MakeBounds(x, y, imgshape, FOVSize):
    ### Given a width and height, and an image size, make sure a box is within the image, and return the bounding region.

    # Make sure the FOV is divisible by 2.
    FOVRadius = int(FOVSize/2)
    assert(FOVRadius*2 == FOVSize)

    # Remember x and y for the image are y and x as reported by the coordinate file.
    mmin, mmax = y - FOVRadius, y + FOVRadius
    nmin, nmax = x - FOVRadius, x + FOVRadius

    # If this x,y coordinate would select a square (radius FOVSize) outside the
    # image, then simply push the square to stay within the image.
    if y < FOVRadius:
        mmin, mmax = 0, FOVRadius * 2
    if x < FOVRadius:
        nmin, nmax = 0, FOVRadius * 2
    if x > imgshape[1] - FOVRadius:
        nmin, nmax = imgshape[1] - 2 * FOVRadius, imgshape[1] + 1
    if y > imgshape[0] - FOVRadius:
        mmin, mmax = imgshape[0] - 2 * FOVRadius, imgshape[0] + 1

    return mmin, mmax, nmin, nmax


def ChopImage(img, FOVSize):
    SubImages = []
    SubImageCentroids = []
    SubImageStride = 5  # Subimages will have their centroids in a grid with this spacing.

    # Make a grid of points which are the centroids for all the sub images.
    xvals = range(FOVSize,
                  img.shape[1]-FOVSize+SubImageStride, # Go as far to the edge as possible -- there may be a little extra overlap on a couple images.
                  SubImageStride)
    yvals = range(FOVSize,
                  img.shape[0]-FOVSize+SubImageStride,
                  SubImageStride)

    # Extract all those images.
    for y in yvals:
        for x in xvals:
            mmin, mmax, nmin, nmax = MakeBounds(x,y, img.shape)
            SubImages.append(img[mmin:mmax, nmin:nmax])
            SubImageCentroids.append((x,y))

    # Turn the list of subimages into an input for the CNN.
    SubImages = np.array(SubImages).reshape(len(SubImages),FOVSize*2,FOVSize*2,1)
    SubImageCentroids = np.array(SubImageCentroids).reshape(len(SubImageCentroids), 2)

    return(SubImages, SubImageCentroids)


def TransformImage(grayscale, alpha, scale=0.5, aspect=1.0, rotate=0.0, shift=(0.0,0.0), FOVSize=30):
    ### TransformImage(grayscale, alpha, scale=0.5, aspect=1.0, rotate=0.0, shift=(0.0,0.0), FOVSize=30)
    #   grayscale: MxN image of the crater.    Values between 0,1.
    #   alpha: MxN alpha mask, png style.  Values between 0,1.
    #   scale: Size of the resulting crater in units of FOVSize.  0 = min size diameter (5 pixels), 1 = FOVsize diameter.
    #   aspect: 1 means no change in aspect ratio.  0.5 means one direction is squished by half relative to the other.  2, stretched by double.
    #   rotate: CCW rotation in degrees.
    #   shift: (x,y) shift after other transformations in units of FOVSize
    #   FOVSize: How many pixels the target FOV is.

    # Note the true center of the image.
    Center = np.array(alpha.shape) / 2

    # Find the center of mass and find the distances of pixels from the COM weighted by alpha (how big is this crater)
    COM = meas.center_of_mass(alpha)
    M, N = np.meshgrid(range(alpha.shape[0]), range(alpha.shape[1]))
    M = M.T
    N = N.T
    dist = np.sqrt((M - COM[0]) ** 2 + (N - COM[1]) ** 2) * alpha

    # Our scale input can be between 0 and 1.  0 means the crater will be shrunk to a minimum size.  1 means it is as big as the FOV.
    MinSize = 5  # 5 pixel minimum size.
    scale = scale*(FOVSize-MinSize) + MinSize  # Scale=0 -> Scale=MinSize.  Scale=1 -> Scale=FOVSize
    # But the crater image could be 100 pixels wide, or 300 or whatever.  We need to put it on the FOV length scale..
    CurrSize = 2*np.max(dist)  # This is the diameter to encompass the outermost portion of the crater.
    scale /= CurrSize

    # Aspect ratio is a perturbation around 1.  1.1 means it is a little taller.  0.9 means it is a little wider.
    zoom = scale*np.array([aspect/1.0, 1.0/aspect])

    # Shift is an M, N tuple which can move the crater in units of FOVSize.
    # The input for an unshifted result is (0,0).  To put it one corner could be (-0.5, -0.5).  To move it completely out of view: (1,1)
    shift = np.array(shift)*FOVSize

    # Move the crater to the exact middle.
    a1 = ndi.shift(alpha,Center-COM)
    # Scale the crater
    a2 = ndi.zoom(a1, zoom)
    # If the crater is zoomed so small that it less than FOVSize, then pad the array back up again.
    if np.any(np.array(a2.shape) < FOVSize):
        padding = np.array(((np.floor((FOVSize - a2.shape[0]) / 2), np.ceil((FOVSize - a2.shape[0]) / 2)), (np.floor((FOVSize - a2.shape[1]) / 2), np.ceil((FOVSize - a2.shape[1]) / 2)))).astype('int')
        padding[padding < 0] = 0 # If one of the axes was > FOVSize we don't want to negative pad it!
        a2 = np.pad(a2, padding, mode='constant')
    # rotate the crater
    a3 = ndi.rotate(a2, rotate)
    # Apply a shift to put the crater not in the center anymore.
    a4 = ndi.shift(a3, shift)

    # Same for the grayscale image.
    g1 = ndi.shift(grayscale,Center-COM)
    g2 = ndi.zoom(g1, zoom)
    if np.any(np.array(g2.shape) < FOVSize):
        padding = np.array(((np.floor((FOVSize - g2.shape[0]) / 2), np.ceil((FOVSize - g2.shape[0]) / 2)), (np.floor((FOVSize - g2.shape[1]) / 2), np.ceil((FOVSize - g2.shape[1]) / 2)))).astype('int')
        padding[padding < 0] = 0
        g2 = np.pad(g2, padding, mode='constant')
    g3 = ndi.rotate(g2, rotate)
    g4 = ndi.shift(g3, shift)

    # Finally, crop it to a single FOV.
    NewCenter = np.round(np.array(a4.shape)/2).astype('int')
    mmin, mmax, nmin, nmax = MakeBounds(NewCenter[1], NewCenter[0], a4.shape, FOVSize)
    a4 = a4[mmin:mmax, nmin:nmax]
    g4 = g4[mmin:mmax, nmin:nmax]

    # The user gets back the grayscale masked by the alpha, and the alpha.
    return g4*a4, a4

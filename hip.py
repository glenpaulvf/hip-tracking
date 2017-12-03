import cv2
from matplotlib import pyplot as plt


# Comparison methods
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


# Template matching in frame
frame = cv2.imread('frame.png', cv2.IMREAD_GRAYSCALE)
frame_copy = frame.copy()

hip_tmpl = cv2.imread('hip.png', cv2.IMREAD_GRAYSCALE)
hip_tmpl_w, hip_tmpl_h = hip_tmpl.shape[::-1]

## Apply template matching per method
for m in methods:
    frame = frame_copy.copy()
    method = eval(m)
    
    # Template matching
    res = cv2.matchTemplate(frame, hip_tmpl, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Take minimum when TM_SQDIFF or TM_SQDIFF_NORMED
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + hip_tmpl_w, top_left[1] + hip_tmpl_h)
    
    cv2.rectangle(frame, top_left, bottom_right, 0, 2)
    
    plt.subplot(121), plt.imshow(res, cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(frame, cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(m)
    
    plt.show()

    ## The resulting plots show which methods pass:
    ## TM_CCOEFF fails
    ## TM_CCOEFF_NORMED passes
    ## TM_CCORR fails
    ## TM_CCORR_NORMED passes
    ## TM_SQDIFF passes
    ## TM_SQDIFF_NORMED passes


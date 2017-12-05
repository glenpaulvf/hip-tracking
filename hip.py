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


# Template matching in video
video = cv2.VideoCapture('RyanRun.mp4')

while(video.isOpened()):
    ret, frame = video.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Template matching using TM_CCOEFF_NORMED
    method = eval('cv2.TM_CCOEFF_NORMED')
     
    res = cv2.matchTemplate(gray, hip_tmpl, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    top_left = max_loc
    bottom_right = (top_left[0] + hip_tmpl_w, top_left[1] + hip_tmpl_h)
    
    cv2.rectangle(gray, top_left, bottom_right, 0, 2)
        
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
cv2.waitKey(1) # Ensure window is destroyed

## Testing playback show which methods pass:
## TM_CCOEFF n/a
## TM_CCOEFF_NORMED passes, minor glitches before hip on desk, block
## TM_CCORR n/a
## TM_CCORR_NORMED fails, major glitches before, during hip on pants, desk
## TM_SQDIFF fails, major glitches before, during hip on bag
## TM_SQDIFF_NORMED fails, major glitches before, during hip on bag


# Template matching in video with initial grabbing
video = cv2.VideoCapture('RyanRun.mp4')

## Store x, y coordinates of hip
pos = []

## Grab first 870 frames
for f in range(0, 871):
    video.grab()
    
## Retrieve next frames
for f in range(871, 1083): # Stop after frame 1082
    ret, frame = video.retrieve()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Template matching using TM_CCOEFF_NORMED
    method = eval('cv2.TM_CCOEFF_NORMED')
     
    res = cv2.matchTemplate(gray, hip_tmpl, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    top_left = max_loc
    pos.append(top_left) # Store x, y coordinaes of hip
    bottom_right = (top_left[0] + hip_tmpl_w, top_left[1] + hip_tmpl_h)
    
    cv2.rectangle(gray, top_left, bottom_right, 0, 2)
        
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    video.grab()
    
video.release()
cv2.destroyAllWindows()
cv2.waitKey(1) # Ensure window is destroyed


# Plot x, y coordinates of hip

## Negate y-values
pos = [(x, -y) for x, y in pos]

## Decouple values
pos = zip(*pos)

## Plot
plt.plot(pos[0], pos[1])
plt.gca().set_xlim(0, 700)
plt.gca().set_ylim(-65, -110)
plt.gca().invert_yaxis() # Invert y-axis
plt.title('Hip Trace')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.show()

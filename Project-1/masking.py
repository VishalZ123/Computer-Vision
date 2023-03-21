import cv2
import numpy as np

def draw_mask(image):
    '''
        Function to draw a custom mask for the source image.
    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Create a window to display the image
    cv2.namedWindow('Draw mask')
    drawing = False
    ix, iy = -1, -1
    curve_pts = []

    # Define a function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        global drawing, ix, iy, curve_pts
        # If the left mouse button is pressed, start drawing a mask
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            curve_pts = [(ix, iy)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                    cv2.line(image, curve_pts[-1], (x, y), (0, 0, 255), 2)
                    curve_pts.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(image, curve_pts[-1], (x, y), (0, 0, 255), 2)
            curve_pts.append((x, y))
            pts = np.array(curve_pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 255))

    # Initialize the mask as a black image with the same size as the input image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Register the mouse callback function
    cv2.setMouseCallback('Draw mask', mouse_callback)

    # Display the image and wait for the user to draw the mask
    while True:
        cv2.imshow('Draw mask', image)
        key = cv2.waitKey(1)

        # If the user presses the 'c' key, clear the mask
        if key == ord('c'):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            image = image.copy()

        # If the user presses the 's' key, save the mask and exit the loop
        elif key == ord('s'):
            cv2.destroyWindow('Draw mask')
            return cv2.merge([mask, mask, mask])

        # If the user presses the 'q' key or closes the window, exit the loop
        elif key == ord('q') or key == 27 or cv2.getWindowProperty('Draw mask', cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow('Draw mask')
            return None
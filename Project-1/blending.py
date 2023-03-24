import numpy as np
import cv2, sys
import matplotlib.pyplot as plt
from masking import draw_mask

def blend(img_dest, img_src, mask, kernel=(1, 1)):
  '''
    Uses Alpha Blending, put the object from source image on the destination image using a mask after applying the gaussain blur using a kernel. \n
    Parameters: \n
        img_dest - Destination Image \n
        img_src - Source Image \n
        mask - Mask of the object from the source to be blended on the destination image \n
        kernel - default - (1, 1) : The kernel for gaussian blur, 1x1 gives simple cut paste
  '''

  smoothed_mask = cv2.blur(mask, kernel)
  normalized_mask = smoothed_mask/255

  img1_float = img_dest.astype(float)
  img2_float = img_src.astype(float)

  image = normalized_mask*img2_float + (1-normalized_mask)*img1_float
  image = image.astype(np.uint8)

  return image, smoothed_mask



class MultiBandBlending():
    def __init__(self, depth=5):
        self.gaussian_pyramid1 =[]
        self.gaussian_pyramid2 =[]
        self.laplacian_pyramid1 = []
        self.laplacian_pyramid2 = []
        self.mask_gaussian_pyramid = []
        self.depth = depth
    
    def quantize_image_shape(self, image):
        new_shape = (image.shape[1] & ~(2**(self.depth-1)-1), image.shape[0] & ~(2**(self.depth-1)-1))
        return  cv2.resize(image, new_shape)
    
    def build_gaussian_pyramid(self, image):
        gaussian_pyramid = [image]
        for _ in range(self.depth-1):
            image = cv2.pyrDown(image)
            gaussian_pyramid.append(image.astype(np.uint8))

        return gaussian_pyramid
    
    def build_laplacian_pyramid(self, gaussian_pyramid):
        laplacian_pyramid = [gaussian_pyramid[-1]]
        for i in range(self.depth-1, 0, -1):
            gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
            laplacian = cv2.subtract(gaussian_pyramid[i-1], gaussian_expanded)
            laplacian_pyramid.append(laplacian.astype(np.uint8))
        
        return laplacian_pyramid

    def blend(self, image1, image2, mask):
        # resize the image & mask to have edge length as a power of 2
        image1 = self.quantize_image_shape(image1)
        image2 = self.quantize_image_shape(image2)
        
        mask = self.quantize_image_shape(mask)

        # build gaussain pyramid of image & mask
        self.gaussian_pyramid1 = self.build_gaussian_pyramid(image1)
        self.gaussian_pyramid2 = self.build_gaussian_pyramid(image2)
        self.mask_gaussian_pyramid = self.build_gaussian_pyramid(mask)

        # build laplacian pyramid of image
        self.laplacian_pyramid1 = self.build_laplacian_pyramid(self.gaussian_pyramid1)
        self.laplacian_pyramid2 = self.build_laplacian_pyramid(self.gaussian_pyramid2)


        blended_pyramid = []
        for i in range(self.depth-1, -1, -1):
            mask_gaussian  = self.mask_gaussian_pyramid[i]/255
            mask_gaussian = mask_gaussian / np.max(mask_gaussian)
            im1 = self.laplacian_pyramid1[self.depth-i-1].astype(float)
            im2 = self.laplacian_pyramid2[self.depth-i-1].astype(float)
            k1 = np.multiply(im1, mask_gaussian).astype(np.uint8)
            k2 = np.multiply((1-mask_gaussian), im2).astype(np.uint8)

            level_blend = k1.astype(np.uint8) + k2.astype(np.uint8)
            blended_pyramid.append(level_blend.astype(np.uint8))


        blended_image = self.reconstruct_image(blended_pyramid)
        
        return blended_image

    def reconstruct_image(self, blended_pyramid):
        blended_image = blended_pyramid[0]
        for i in range(1, len(blended_pyramid)):
            blended_image = cv2.add(cv2.pyrUp(blended_image), blended_pyramid[i])
        return blended_image

def poisson_blending(image1, image2, mask, p):
    return cv2.seamlessClone(image2, image1, mask, p, cv2.NORMAL_CLONE)

if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print("Usage: python blending.py source_path dest_path method_name [mask_path]")
        sys.exit()

    source_path = sys.argv[1]
    dest_path = sys.argv[2]
    method_name = sys.argv[3]

    mask_path = None
    mask_img = None
    if len(sys.argv) >= 5:
        mask_path = sys.argv[4]

    # Read the source and destination images in BGR format
    src_img = cv2.imread(source_path)
    dest_img = cv2.imread(dest_path)

    # Convert the images to RGB format
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2RGB)

    # Check if mask_path is provided
    if mask_path is not None:
        mask_img = cv2.imread(mask_path)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    else:
        mask_img = draw_mask(src_img)
    
    blended_img = None
    if method_name == 'cut_paste':
        blended_img, _ = blend(dest_img, src_img, mask_img)
    elif method_name == 'alpha':
        print('Using Gaussian blur with kernel kxk\nEnter k: ')
        k = int(input())
        blended_img, _ = blend(dest_img, src_img, mask_img, kernel=(k,k))
    elif method_name == 'multi_band':
        blender = MultiBandBlending(depth=6)
        blended_img = blender.blend(src_img, dest_img, mask_img)
    elif method_name == 'poisson':
        print('Enter coordinates to blend the source-')
        print('Enter x-')
        x = int(input())
        print('Enter y-')
        y = int(input())
        blended_img = poisson_blending(src_img, dest_img, mask_img, (x, y))
    else:
        print("Invalid method name")
        exit()
    
    plt.imsave('blended.jpg', blended_img)
    plt.imshow(blended_img)
    plt.show()
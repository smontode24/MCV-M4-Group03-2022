# vanishing points
import vp_detection as vp

import utils as utils
import matplotlib.pyplot as plt
import cv2

def estimate_vps(img, num_img=0, show_vp=True):
    # Find the vanishing points of the image, through Xiaohulu methoed
    length_thresh = 35 #20  # Minimum length of the line in pixels
    # length_thresh 45 in lab lab5-GroupN
    seed = 1 # Or specify whatever ID you want (integer) Ex: 1337
    img = img.copy()
    vpd = vp.VPDetection(length_thresh=length_thresh, seed=seed, principal_point=(456.21, 302.04), focal_length=(708.64 + 709.84)/2.0)
    vps = vpd.find_vps(img)

    if utils.debug >= 0:
        print("  Vanishing points found")
    if utils.debug > 1:
        print("      vps coordinates:\n", vpd.vps_2D)
    if utils.debug > 2:
        print("      length threshold:", length_thresh)
        print("      principal point:", vpd.principal_point)
        print("      focal length:", vpd.focal_length)
        print("      seed:", seed)
        
    if num_img == 1:
        i1, i2 = 0, 1
        tmp_x, tmp_y = vpd.vps_2D[i1][0], vpd.vps_2D[i1][1]
        vpd.vps_2D[i1][0] = vpd.vps_2D[i2][0]
        vpd.vps_2D[i1][1] = vpd.vps_2D[i2][1]
        vpd.vps_2D[i2][0] = tmp_x
        vpd.vps_2D[i2][1] = tmp_y
        
        #vpd.vps_2D[0][0] = -1 * vpd.vps_2D[0][0] * (100000)
        #vpd.vps_2D[0][1] = -1 * vpd.vps_2D[0][1] * (100000)
        
        #vpd.vps_2D[2][0] = -1 * vpd.vps_2D[2][0]*(10e+9)
        #vpd.vps_2D[2][1] = -1 * vpd.vps_2D[2][1]*(10e+9)
        
    if show_vp:
        img_v = vpd.create_debug_VP_image()
        img_v = cv2.line(img_v, (int(img_v.shape[1])//2, int(img_v.shape[0])//2), (int(vpd.vps_2D[0][0]), int(vpd.vps_2D[0][1])), (255, 0, 0), 2)
        img_v = cv2.line(img_v, (int(img_v.shape[1])//2, int(img_v.shape[0])//2), (int(vpd.vps_2D[1][0]), int(vpd.vps_2D[1][1])), (0, 255, 0), 2)
        img_v = cv2.line(img_v, (int(img_v.shape[1])//2, int(img_v.shape[0])//2), (int(vpd.vps_2D[2][0]), int(vpd.vps_2D[2][1])), (0, 0, 255), 2)
        plt.imshow(img_v)
        plt.title("Vanishing lines")
        plt.show()
    
    lines, clusters = vpd.get_lines_and_clusters()
    lines_vp = []
    for i in range(3):
        lines_vp_i = lines[clusters[i]]
        lines_vp.append(lines_vp_i)
    return vpd.vps_2D, lines_vp

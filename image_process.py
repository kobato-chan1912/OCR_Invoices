import numpy as np
import cv2 # CV2 phải lớn hơn 3.0.5
import matplotlib.pyplot as plt

from skimage.filters import threshold_local
from PIL import Image

# Bước 1: Nhận dạng vật thể hoá đơn thông qua xử lý lọc đường viền 

# Sample file out of the dataset
file_name = 'images/input.png'
img = Image.open(file_name)
img.thumbnail((800,800), Image.ANTIALIAS)
img # In ảnh 

# Tiền xử lý ảnh

# Hàm Resize ảnh 
def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# Hàm chuyển ảnh về gray 

def plot_gray(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(image, cmap='Greys_r')

def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# Đọc ảnh
image = cv2.imread(file_name)
# Downscale image as finding receipt contour is more efficient on a small image
resize_ratio = 500 / image.shape[0]
original = image.copy()
image = opencv_resize(image, resize_ratio)



# Chuyển ảnh sang Grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plot_gray(gray) # In ảnh grayscale 

# Dùng GaussianBlur 5x5 để giảm nhiễu 
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
plot_gray(blurred)

# Hiển thị các Object trong hình dạng cạnh viền (nền đen)

edged = cv2.Canny(dilated, 100, 200, apertureSize=3)
plot_gray(edged)


# Detect tất cả các cạnh viền của vật thể 
contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)
plot_rgb(image_with_contours)

# Giữ lại 10 đường viền rõ nét 
largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
image_with_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0,255,0), 3)
plot_rgb(image_with_largest_contours)


# Chuẩn hoá các cạnh về đa giác
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

def get_receipt_contour(contours):    
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # Nếu đường cạnh của đa giác đã chuẩn hoá  = 4 => Hình chữ nhật của biên lai 
        if len(approx) == 4:
            return approx

get_receipt_contour(largest_contours)

receipt_contour = get_receipt_contour(largest_contours)
image_with_receipt_contour = cv2.drawContours(image.copy(), [receipt_contour], -1, (0, 255, 0), 2)
plot_rgb(image_with_receipt_contour)

# Bước 2: Cắt xén vật thể (Scan vật thể)

# Chuyển đổi các đường viền (countor) thành một mảng toạ độ
# Dùng các rectangle points của mảng trên để tính toán destination points của các view đã quét 
# Tính toán ma trận chuyển đổi bằng cv2.getPerspectiveTransform
# Khôi phục phối cảnh bằng cv2.warpPerspective 


def contour_to_rect(contour):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio



def wrap_perspective(img, rect):
    # unpack rectangle points: top left, top right, bottom right, bottom left
    (tl, tr, br, bl) = rect
    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # destination points which will be used to map the screen to a "scanned" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # warp the perspective to grab the screen
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour))
plt.figure(figsize=(16,10))
plt.imshow(scanned)


def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255

result = bw_scanner(scanned)
plot_gray(result)

output = Image.fromarray(result)
output.save('result.png')

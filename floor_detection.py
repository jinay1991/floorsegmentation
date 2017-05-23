import cv2
import numpy as np
import os
import logging
import random
from scipy.spatial import distance as dist

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

class FloorSegmentation():

    def __init__(self, filename):
        self.bgr_image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.bgr_image is None:
            log.error("can not open file")
            exit(1)

        # convert to RGB from BGR
        self.rgb_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)

        blur = cv2.bilateralFilter(self.rgb_image, 9, 30, 30)

        self.gray_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        (self.height, self.width) = self.gray_image.shape[:2]

        if log.level == logging.DEBUG:
            plt.subplot(131), plt.imshow(self.rgb_image), plt.title("rgb")
            plt.subplot(132), plt.imshow(blur), plt.title("blur - GaussianBlur")
            plt.subplot(133), plt.imshow(self.gray_image, cmap='gray'), plt.title("gray")
            plt.show()

    def threshold(self):
        """
        Generates threshold Image from Input Image
        """

        ret, self.thresh_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        log.debug("threshold value: %s", ret)

        # comput Surface Score
        ret, contours, hierarchy = cv2.findContours(self.thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        def get_nearest_neighbour(test_pt, query_set_pts):
            distances = []
            for m in range(0, len(query_set_pts)):
                for x in range(0, len(query_set_pts[m])):
                    distance = dist.euclidean(test_pt, query_set_pts[m][x])
                    distances.append((query_set_pts[m][x], distance))
            distances.sort(key=lambda x: x[1], reverse=True)
            if len(distances) > 0:
                return distances[0][0]
            else:
                return 0

        line_image = np.zeros(self.thresh_image.shape[:2], dtype=np.uint8)
        structure_score = 0.0
        for (pt1, pt2) in self.HorizontalLines:
            cv2.line(line_image, pt1, pt2, 255, 1)
            _, line_contour,_ = cv2.findContours(line_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            nearest_pt1 = get_nearest_neighbour(pt1, line_contour)
            # log.debug("nearest point1 %s", nearest_pt1)
            structure_score += dist.euclidean(pt1, nearest_pt1)

        log.info("structure_score: %s", structure_score)
        sigmaB = 30
        norm_score = np.exp(-(structure_score)/(2 * sigmaB * sigmaB))
        log.info("Normalized Score: %s", norm_score)

        if log.level == logging.DEBUG:
            plt.imshow(self.thresh_image, cmap='gray'), plt.title("threshold")
            plt.show()

        phi_s = norm_score
        return phi_s

    def line_segments(self):
        """
        Generates canny edge detection from the original input image
        Canny Edge Detection -> Robust Line Fitting -> Pruning Line Segments

        Results: vertical line segments and horizontal line segments
        """

        # ---------------------
        # Canny Edge Detection
        # ---------------------
        self.edges_image = cv2.Canny(self.gray_image, 40, 80, apertureSize=3)

        # --------------------
        # Robust Line Fitting
        # --------------------
        # - Step-1 Douglas Piere Algorithm [cv2.approxPolyDP]
        #    - Modification applied from _ref11.pdf
        ret, contours, hierarchy = cv2.findContours(self.edges_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(c, epsilon=0.1*cv2.arcLength(c,closed=True), closed=True) for c in contours]

        lineH = []
        lineV = []

        lines = cv2.HoughLinesP(
                self.edges_image, 1, np.pi / 2, 10, None, minLineLength=15, maxLineGap=10)
        if lines is not None:
            for [[x1, y1, x2, y2]] in lines:
                angle = round(np.arctan2(y2 - y1, x2 - x1)
                                * 180. / np.pi, 2)
                # log.debug("angle: {} degree".format(angle))
                if angle < 45 and angle > -45:  # Horizontal Lines
                    # 45 degree +/- from horizontal Axes
                    lineH.append(((x1, y1), (x2, y2)))
                elif (angle > 85 and angle < 95) or (angle > -95 and angle < -85):  # Vertical Lines
                    # 5 degree +/- from Vertical Axes
                    lineV.append(((x1, y1), (x2, y2)))
                else:
                    continue
        else:
            log.error("no lines found with HoughLineP")
            return None, None

        # ----------------------
        # Pruning Line Segments
        # ----------------------
        # thresholding lines if <15pix for Horizontal and <60pix for Vertical, discard them.
        early_countH = len(lineH)
        early_countV = len(lineV)
        for (pt1, pt2) in lineH:
            distance = dist.euclidean(pt1, pt2)
            # log.debug("distance between %s and %s is %s", pt1, pt2, distance)
            if distance < 15:
                lineH.remove((pt1, pt2))
        for (pt1, pt2) in lineV:
            distance = dist.euclidean(pt1, pt2)
            # log.debug("distance between %s and %s is %s", pt1, pt2, distance)
            if distance < 60:
                lineV.remove((pt1, pt2))

        log.info("Total Horizontal lines removed by thresh=15pix: %s", early_countH - len(lineH))
        log.info("Total vertical lines removed by thresh=60pix: %s", early_countV - len(lineV))

        # compute a, b and c of line equation (ax + by + c = 0)
        lineConstH = []
        lineConstV = []
        for ((x1, y1), (x2, y2)) in lineH:
            a = y1 - y2
            b = x2 - x1
            c = (-1) * ((x1 * y2) - (x2 * y1))
            lineConstH.append([a,b,c])
        for ((x1, y1), (x2, y2)) in lineV:
            a = y1 - y2
            b = x2 - x1
            c = (-1) * ((x1 * y2) - (x2 * y1))
            lineConstV.append([a,b,c])

        def intersection(l1, l2):
            [a1, b1, c1] = l1
            [a2, b2, c2] = l2
            D = a1 * b2 - b1 * a2
            Dx = c1 * b2 - b1 * c2
            Dy = a1 * c2 - c1 * a2
            if D != 0:
                intersection_pt = (Dx / D, Dy / D)
                return intersection_pt
            else:
                return None
        # compute intersection points for pair of Horizontal Lines
        intersectionPtsH = []
        y_values = np.empty(1,dtype=np.uint8)
        for i in range(0, len(lineH)):
            for j in range(i, len(lineH)):
                pt = intersection(lineConstH[i], lineConstH[j])
                if pt is not None:
                    intersectionPtsH.append(pt)
                    np.append(y_values, pt[1])
                    log.debug("Found Horizontal line intersection point at %s", pt)

        log.debug("Total Intersection points found: %s", len(intersectionPtsH))
        y_mean = np.mean(y_values)
        log.info("mean Y value for horizontal lines: %s", y_mean)

        # vanishing line is at (x_min, y_mean) -> (x_max, y_mean), discard lines lies above this
        for (pt1, pt2) in lineH:
            if pt1[1] < y_mean or pt2[1] < y_mean:
                lineH.remove((pt1, pt2))
        for (pt1, pt2) in lineV:
            if pt1[1] < y_mean or pt2[1] < y_mean:
                lineV.remove((pt1, pt2))

        # remove all those vertical lines which has bottom point above h/2 of image
        for (pt1, pt2) in lineV:
            if pt1[1] < pt2[1]:
                if pt2[1] > self.height / 2:
                    continue
                else:
                    lineV.remove((pt1,pt2))
            else:
                if pt1[1] > self.height / 2:
                    continue
                else:
                    lineV.remove((pt1,pt2))
        import operator

        lineV.sort(key=operator.itemgetter(0))

        bottom_pts = []
        for (pt1, pt2) in lineV:
            if pt1[1] > pt2[1]:
                bottom_pts.append(pt2)
            else:
                bottom_pts.append(pt1)

        # bottom_pts = cv2.approxPolyDP(np.asarray(bottom_pts),epsilon=2.5, closed=True)

        # ploting things if in debug
        if log.level == logging.DEBUG:
            rows, cols = self.edges_image.shape[:2]
            vis_raw = np.zeros((rows, cols, 3), dtype=np.uint8) # 3 Channel Color Image
            vis_in = self.rgb_image.copy()

            # for (pt1, pt2) in lineH:
            #     cv2.line(vis_in, pt1, pt2, (255, 255, 0), thickness=2)
            #     cv2.line(vis_raw, pt1, pt2, (255, 255, 0), thickness=2)
            # for (pt1, pt2) in lineV:
            #     cv2.line(vis_in, pt1, pt2, (255, 0, 255), thickness=2)
            #     cv2.line(vis_raw, pt1, pt2, (255, 0, 255), thickness=2)
            for i in range(0, len(bottom_pts) - 1):
                pt1 = (bottom_pts[0][i][0][0], bottom_pts[0][i][0][1])
                pt2 = (bottom_pts[0][i + 1][0][0], bottom_pts[0][i + 1][0][1])
                cv2.line(vis_raw, pt1, pt2, (0, 0, 255), thickness=1)
                cv2.line(vis_in, pt1, pt2, (255, 0, 255), thickness=2)

            cv2.drawContours(vis_raw, bottom_pts, -1, (0, 0, 255), thickness=2)

            plt.subplot(131), plt.imshow(self.edges_image, cmap='gray'), plt.title("canny")
            plt.subplot(132), plt.imshow(vis_raw), plt.title("vis_raw")
            plt.subplot(133), plt.imshow(vis_in), plt.title("vis_in")
            plt.show()

        # results
        self.HorizontalLines = lineH
        self.VerticalLines = lineV

        phi_b = 0
        return phi_b


    def run(self):
        pass

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file", required=True)
    parser.add_argument("--debug", help="debug enable", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(os.path.abspath(args.input)):
        log.error("No file found at given path")
        exit(1)

    if args.debug:
        from matplotlib import pyplot as plt
        log.setLevel(logging.DEBUG)

    floor_seg = FloorSegmentation(args.input)

    floor_seg.line_segments()
    # floor_seg.threshold()


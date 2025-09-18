import pandas as pd
import random
import math


class RANSAC:
    def __init__(self, point_cloud, max_iterations,            
                 distance_ratio_threshold):
        self.point_cloud = point_cloud
        self.max_iterations = max_iterations
        self.distance_ratio_threshold = distance_ratio_threshold

    def _ransac_algorithm(self):

        inliers_result = set()
        while self.max_iterations:
            self.max_iterations -= 1

            random.seed()
            inliers = []
            while len(inliers) < 3:
                random_index = random.randint(0, len(self.point_cloud)-1)
                inliers.append(random_index)

            x1, y1, z1 = self.point_cloud.loc[inliers[0]]
            x2, y2, z2 = self.point_cloud.loc[inliers[1]]
            x3, y3, z3 = self.point_cloud.loc[inliers[2]]

            #value of constants
            a = (y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1)
            b = (z2 - z1)*(x3 - x1) - (x2 - x1)*(z3 - z1)
            c = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
            d = -(a*x1 + b*y1 + c*z1)
            plane_lenght = max(0.1, math.sqrt(a*a + b*b + c*c))

            for point in self.point_cloud.iterrows():
                index = point[0]

                if index in inliers:
                    continue

                x, y, z = point[1]
                # distance of point from the plane
                distance = math.fabs(a*x + b*y + c*z + d)/plane_lenght
                # check if itis an inlier
                if distance <= self.distance_ratio_threshold:
                    inliers.append(index)
            # check if this plane contains most number of points
            if len(inliers) > len(inliers_result):
                inliers_result.clear()
                inliers_result = inliers

        inlier_points = pd.DataFrame(columns=["X", "Y", "Z"])
        outlier_points = pd.DataFrame(columns=["X", "Y", "Z"])
        for point in self.point_cloud.iterrows():
            if point[0] in inliers_result:
                '''inlier_points = pd.concat([inlier_points, pd.DataFrame([{"X": point[1]["X"],
                                                                         "Y": point[1]["Y"],
                                                                         "Z": point[1]["Z"]}])], ignore_index=True)'''
                inlier_points.loc[len(inlier_points)]=[point[1]["X"], point[1]["Y"], point[1]["Z"]]
                continue
            '''outlier_points = pd.concat([outlier_points, pd.DataFrame([{"X": point[1]["X"],
                                                                       "Y": point[1]["Y"],
                                                                       "Z": point[1]["Z"]}])], ignore_index=True)'''
            outlier_points.loc[len(outlier_points)]=[point[1]["X"], point[1]["Y"], point[1]["Z"]]

        return inlier_points, outlier_points

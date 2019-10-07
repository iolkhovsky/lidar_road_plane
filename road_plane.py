import numpy as np


class RandomSampleRegressor:

    def __init__(self):
        self.cloud = None  # np.ndarray of spatial (x,y,z) points
        self.points_cnt = None
        self.threshold = None  # criterion for evaluating belonging to a plane (maximum distance from plane to point)
        # additional settings
        self.iter_limit = None  # limit for plane search iterations (if None or <1 calculated automatically)
        self.inlier_prob = 0.5  # probability of inlier selection at each iteration
        self.success_prob = 0.99  # probability of success (correct plane estimation)
        # output data
        self.plane = None  # tuple of plane a,b,c,d coefficients
        self.capacity = None  # share of points, which self.plane contains for sure; can be less (enough for inlier_p)
        pass

    def config(self, thresh, iter_max=-1, inlier_p=0.5, success_p=0.99):
        self.threshold = thresh
        if iter_max > 1:  # if got correct iteration max cnt
            self.iter_limit = iter_max
        else:  # otherwise automatically estimate max iterations cnt
            self.inlier_prob = inlier_p
            self.success_prob = success_p
            self.iter_limit = int(np.log10(1-self.success_prob) / np.log10(1-np.power(self.inlier_prob, 3)))
            if self.iter_limit < 1:
                self.iter_limit = 1
        pass

    def fit(self, cloud):
        self.cloud = cloud
        self.points_cnt = self.cloud.shape[0]
        ready = False
        it_cnt = 0
        while not ready:
            # choose 3 random points from cloud
            p1id = np.random.randint(0, self.points_cnt)
            p2id = np.random.randint(0, self.points_cnt)
            p3id = np.random.randint(0, self.points_cnt)
            if self.__invalid_points(self.cloud[p1id], self.cloud[p2id], self.cloud[p3id]):
                it_cnt += 1
                continue
            # define plane by 3 selected points
            self.plane = self.__get_plane_from_points(self.cloud[p1id], self.cloud[p2id], self.cloud[p3id])
            # compute share of points in cloud, belong to current plane
            self.capacity = self.__get_capacity()
            it_cnt += 1
            if self.capacity >= self.inlier_prob or it_cnt > self.iter_limit:
                ready = True
        pass

    def get_estimated_plane(self):
        return self.plane, self.capacity

    def __get_capacity(self):
        inliers_cnt = 0
        dest_cnt = int(self.inlier_prob * self.points_cnt) + 1
        for i in range(self.points_cnt):
            if self.__point_belongs_to_plane(self.cloud[i]):
                inliers_cnt += 1
            if inliers_cnt >= dest_cnt:
                break
        return float(inliers_cnt) / self.points_cnt

    def __point_belongs_to_plane(self, point):
        return self.__point_to_plane_distance(point, self.plane) <= self.threshold

    @staticmethod
    def __point_to_plane_distance(point, plane):
        vec1 = np.asarray(plane, dtype=np.float64)
        vec2 = np.append(point, 1)
        numerator = np.fabs(np.dot(vec1, vec2))
        denominator = np.sqrt(np.power(plane[0], 2)+np.power(plane[1], 2)+np.power(plane[2], 2))
        if np.fabs(denominator) < 1e-6:
            denominator = 1e-6
        dist = numerator / denominator
        # print("Dist: ", dist)
        return dist

    @staticmethod
    def __invalid_points(p1, p2, p3):
        if np.array_equal(p1, p2) or np.array_equal(p1, p3) or np.array_equal(p2, p3):
            return True
        va = p2 - p1
        vb = p3 - p1
        cross = np.cross(va, vb)
        length = np.linalg.norm(cross)
        if np.fabs(length) < 1e-6:
            return True
        else:
            return False

    @staticmethod
    def __get_plane_from_points(a, b, c):
        ax, ay, az = a[0], a[1], a[2]
        bx, by, bz = b[0], b[1], b[2]
        cx, cy, cz = c[0], c[1], c[2]
        plane_a = (by - ay) * (cz - az) - (cy - ay) * (bz - az)
        plane_b = (bz - az) * (cx - ax) - (cz - az) * (bx - ax)
        plane_c = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        plane_d = -1 * (plane_a * ax + plane_b * ay + plane_c * az)
        norm = max([abs(max([plane_a, plane_b, plane_c, plane_d])), abs(min([plane_a, plane_b, plane_c, plane_d]))])
        k = 1.0 / norm
        plane_a *= k
        plane_b *= k
        plane_c *= k
        plane_d *= k
        return plane_a, plane_b, plane_c, plane_d


class PlaneEstimator:

    def __init__(self):
        # input data
        self.raw_data = None
        self.p = None
        self.points_cnt = None
        self.cloud = None
        self.regressor = RandomSampleRegressor()
        pass

    def load_data(self, path=None):
        if path is not None:
            self.raw_data = self.__read_from_file(path)  # read from file
        else:
            self.raw_data = self.__read_from_console()  # read from console
        self.__parse_data()
        # print("Input data:")
        # print("P = ", self.p)
        # print("Points cnt: ", self.points_cnt)
        # print("Points cloud: ", self.cloud)
        pass

    def estimate_plane(self):
        self.regressor.config(thresh=self.p)
        self.regressor.fit(self.cloud)
        plane, cap = self.regressor.get_estimated_plane()
        # print("Result (plane/cap): ", plane, " / ", cap)

        return plane

    def __parse_data(self):
        # split it into numbers
        words = self.raw_data.split()
        self.p = float(words[0])
        self.points_cnt = int(words[1])
        coord_list = list(map(float, words[2:]))
        self.cloud = np.reshape(np.asarray(coord_list, dtype=np.float64), newshape=(self.points_cnt, 3))
        pass

    @staticmethod
    def __read_from_console():
        outtxt = ""
        # outtxt += input("Input the tolerance (P): ") + "\n"
        outtxt += input() + "\n"
        # cntstr = input("Input points count (at least 3 for plane definition): ") + "\n"
        cntstr = input() + "\n"
        outtxt += cntstr
        cnt = int(cntstr)
        for i in range(cnt):
            # instr = input("Enter point #" + str(i) + " coordinates with spaces as delimiter (x y z) : ")
            instr = input()
            outtxt += instr + "\n"
        return outtxt

    @staticmethod
    def __read_from_file(path):
        # reading text data from file
        text = None
        with open(path, "r") as f:
            text = f.read()
        return text


def main():
    plane_est = PlaneEstimator()  # road plane regressor object
    # input_method = input("Choose input method: stdin(s) or file(f): ")
    # while input_method not in ["s", "f"]:
    #    input_method = input("Choose input method: stdin(s) or file(f): ")
    # if input_method == "f":
    #    print("Loading from text file...")
    #    path = input("Enter full abs path to data-file:")
    #    plane_est.load_data(path)
    # else:
    #    print("Loading from std input...")
    #    plane_est.load_data()
    plane_est.load_data()
    plane = plane_est.estimate_plane()
    print(plane[0], plane[1], plane[2], plane[3])
    # print("Result plane coefficients: ", plane)
    pass


if __name__ == '__main__':
    main()
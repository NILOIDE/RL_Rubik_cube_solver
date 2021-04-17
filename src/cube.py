import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import torch
from bitarray import bitarray
from bitarray.util import ba2int
import cv2


class Cube:

    def __init__(self, n):
        self.n = n
        # self.piece_positions = np.arange(n**3 - (n-2)**3, dtype=np.uint16)
        self.face_positions = np.arange((n-2)**2 * 6, dtype=np.uint16)
        self.edge_positions = np.arange((n-2) * 12, dtype=np.uint16)
        # self.edge_positions[56] = 38
        # self.edge_positions[38] = 56
        self.corner_positions = np.arange(8, dtype=np.uint8)
        # Each corner has 2 bits to mark their 3 potential orientations
        self.corner_orientations = bitarray(0 for _ in range(8 * 2))
        # Each corner has 1 bits to mark their 2 potential orientations
        self.edge_orientations = bitarray(0 for _ in range(12 * (n - 2)))


class CubeVisualizer:
    SIDE_ALIAS = {"F": 0, "B": 2, "D": 1, "U": 3, "L": 4, "R": 5,
                  "Front": 0, "Back": 2, "Down": 1, "Up": 3, "Left": 4, "Right": 5}
    COLOR_ALIAS = {"O": np.array((1.0, .5, 0)),
                   "W": np.array((1.0, 1.0, 1.0)),
                   "R": np.array((1.0, 0, 0)),
                   "Y": np.array((1.0, 1.0, 0)),
                   "G": np.array((0, 1.0, 0)),
                   "B": np.array((0, 0, 1.0)),
                   }
    CORNER_COLORS = {0: "OGY",  # Rotate cube upwards FDBU (faces OWRY), we take the two top corners of every face
                     1: "OYB",  # Color are ordered based on clockwise rotation starting from current face.
                     2: "WGO",
                     3: "WOB",
                     4: "RGW",
                     5: "RWB",
                     6: "YGR",
                     7: "YRB",
                     }

    EDGE_COLORS = {0: "OY",  # Centered on the Front and Back sides
                   1: "OG",
                   2: "OB",
                   3: "WO",
                   4: "WG",
                   5: "WB",
                   6: "RW",
                   7: "RG",
                   8: "RB",
                   9: "YR",
                   10: "YG",
                   11: "YB",
                   }

    def __init__(self, n):
        self.n = n
        self.piece_positions = np.arange(n**3 - (n-2)**3, dtype=np.uint16)
        self.face_c_map = self._create_face_color_map(n)
        self.edge_c_map = self._create_edge_color_map(n)
        self.corner_c_map = self._create_corner_color_map()

    def _create_face_color_map(self, n: int):
        faces_per_side = (n-2)**2
        c_map = np.zeros((faces_per_side * 6, 3))
        for i, c in enumerate(self.COLOR_ALIAS):
            c_map[i*faces_per_side:(i+1)*faces_per_side] = np.tile(self.COLOR_ALIAS[c], (faces_per_side, 1))
        return c_map

    def _create_edge_color_map(self, n: int):
        pieces_per_edge = (n-2)
        edges = 12
        c_map = np.zeros((pieces_per_edge * edges, 2, len("RGB")))
        for i, cs in self.EDGE_COLORS.items():
            piece_c_array = np.stack([self.COLOR_ALIAS[c] for c in cs])
            c_map[i*pieces_per_edge:(i+1)*pieces_per_edge] = np.tile(piece_c_array, (pieces_per_edge, 1, 1))
        return c_map

    def _create_corner_color_map(self):
        corners = 8
        c_map = np.zeros((corners, 3, len("RGB")))
        for i, cs in self.CORNER_COLORS.items():
            piece_c_array = np.stack([self.COLOR_ALIAS[c] for c in cs])
            c_map[i] = piece_c_array
        return c_map

    def visualize(self, faces, edges, e_orientations, corners, c_orientations):
        img = np.zeros((self.n * 6 + 6, self.n + 2, 3))

        n = self.n
        s = n - 2
        a = s**2
        # ######## Corners ###################################################
        """ Do Front->Down->Back->Up face's corners:
            Corners are oriented such that side on position 0 is in {F, D, B, or T}, and go clockwise from there. """
        for c in range(8):
            row = c // 2 * n
            c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
            if c % 2 == 0:
                # If corner is on top-left, corner face orientation will be: - position 0 displayed in [F, D, B, T]
                #                                                            - position 1 displayed in Left face.
                #                                                            - position 2 displayed in [T, F, D, B].
                front, left, top = c_ori, (c_ori + 1) % 3, (c_ori + 2) % 3
                img[row, 0] = self.corner_c_map[corners[c], left]
                img[row+1, 0] = self.corner_c_map[corners[c], left]
                img[row, 1] = self.corner_c_map[corners[c], top]
                img[row+1, 1] = self.corner_c_map[corners[c], front]
            else:
                # If corner is on top-left, corner face orientation will be: - position 0 displayed in [F, D, B, T]
                #                                                            - position 1 displayed in [T, F, D, B].
                #                                                            - position 2 displayed in Right face.
                front, top, right = c_ori, (c_ori + 1) % 3, (c_ori + 2) % 3
                img[row, -1] = self.corner_c_map[corners[c], right]
                img[row+1, -1] = self.corner_c_map[corners[c], right]
                img[row, -2] = self.corner_c_map[corners[c], top]
                img[row+1, -2] = self.corner_c_map[corners[c], front]

        # Loop around the cube by duplicating corners c0 and c1 below for visualization properties
        row = 4 * n
        c_ori = ba2int(c_orientations[0:2])
        front, left, top = c_ori, (c_ori + 1) % 3, (c_ori + 2) % 3
        img[row, 0] = self.corner_c_map[corners[0], left]
        img[row + 1, 0] = self.corner_c_map[corners[0], left]
        img[row, 1] = self.corner_c_map[corners[0], top]
        img[row + 1, 1] = self.corner_c_map[corners[0], front]
        c_ori = ba2int(c_orientations[2:4])
        front, top, right = c_ori, (c_ori + 1) % 3, (c_ori + 2) % 3
        img[row, -1] = self.corner_c_map[corners[1], right]
        img[row + 1, -1] = self.corner_c_map[corners[1], right]
        img[row, -2] = self.corner_c_map[corners[1], top]
        img[row + 1, -2] = self.corner_c_map[corners[1], front]

        """ Do Left face's corners:
            Corners for Left face are always in orientation index 1, but since we are drawing it 
            from a side view we have to add additional rotation based on corner index. """
        corner_idxs = [6, 0, 4, 2]
        for i, c in enumerate(corner_idxs):
            # We account for height of the duplicate corner c0 & c1 => +2
            row = (4 + (i//2)) * n + 2
            if i % 2 == 0:  # If corner is being drawn on the left side of representation
                c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2]) - c//2  # Rotate corner for view display  => c//2
                top, front, side = c_ori % 3, (c_ori + 1) % 3, (c_ori + 2) % 3
                img[row, 0] = self.corner_c_map[corners[c], side]
                img[row+1, 0] = self.corner_c_map[corners[c], side]
                img[row, 1] = self.corner_c_map[corners[c], top]
                img[row+1, 1] = self.corner_c_map[corners[c], front]
            else:  # If corner is being drawn on the left side of representation
                c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2]) - c//2  # Rotate corner for view display  => c//2
                side, front, top = c_ori % 3, (c_ori + 1) % 3, (c_ori + 2) % 3
                img[row, -1] = self.corner_c_map[corners[c], side]
                img[row+1, -1] = self.corner_c_map[corners[c], side]
                img[row, -2] = self.corner_c_map[corners[c], top]
                img[row+1, -2] = self.corner_c_map[corners[c], front]

        """ Do Right face's corners:
            Corners for right face are always in orientation index 2, but since we are drawing it 
            from a side view we have to add additional rotation based on corner index. """
        corner_idxs = [1, 7, 3, 5]
        for i, c in enumerate(corner_idxs):
            # We account for height of the duplicate corners c0 & c1 and Left's drawing = 4
            row = (5 + (i//2)) * n + 4
            if i % 2 == 0:  # If corner is being drawn on the left side of representation
                c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2]) + c//2 + 1
                top, front, side = c_ori % 3, (c_ori + 1) % 3, (c_ori + 2) % 3
                img[row, 0] = self.corner_c_map[corners[c], side]
                img[row+1, 0] = self.corner_c_map[corners[c], side]
                img[row, 1] = self.corner_c_map[corners[c], top]
                img[row+1, 1] = self.corner_c_map[corners[c], front]
            else:  # If corner is being drawn on the left side of representation
                c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2]) + c//2 + 1
                side, front, top = c_ori % 3, (c_ori + 1) % 3, (c_ori + 2) % 3
                img[row, -1] = self.corner_c_map[corners[c], side]
                img[row+1, -1] = self.corner_c_map[corners[c], side]
                img[row, -2] = self.corner_c_map[corners[c], top]
                img[row+1, -2] = self.corner_c_map[corners[c], front]

        # ######## Edges ###################################################
        """ Do Front->Down->Back->Up side's edges:
            Edges are oriented such that for each face in {F, D, B, T}, the upper, left, and right edges
            of a face belong to that face and thus their side of the edge will be on index 0. """
        for e in range(12):
            row = e // 3 * n
            for i in range(s):
                e_idx = e * s + i
                e_ori = int(e_orientations[e_idx])
                if e % 3 == 0:  # Do this cubeface's upper edge
                    front, top = e_ori, (e_ori + 1) % 2
                    img[row, 2+i] = self.edge_c_map[edges[e_idx], top]  # Neighbouring face upwards
                    img[row + 1, 2+i] = self.edge_c_map[edges[e_idx], front]
                elif e % 3 == 1:  # Do this cubeface's left edge
                    front, left = e_ori, (e_ori + 1) % 2
                    img[row+2+i, 0] = self.edge_c_map[edges[e_idx], left]  # Neighbouring face to the left
                    img[row+2+i, 1] = self.edge_c_map[edges[e_idx], front]
                else:  # Do this cubeface's right edge
                    front, right = e_ori, (e_ori + 1) % 2
                    img[row+2+i, -1] = self.edge_c_map[edges[e_idx], right]  # Neighbouring face to the right
                    img[row+2+i, -2] = self.edge_c_map[edges[e_idx], front]

        # Duplicate e0 below for visualization properties
        row = 12 // 3 * n
        for i in range(s):
            e_ori = int(e_orientations[i])
            front, top = e_ori, (e_ori + 1) % 2
            img[row, 2 + i] = self.edge_c_map[edges[i], top]
            img[row + 1, 2 + i] = self.edge_c_map[edges[i], front]

        """ Do Left side's edges:
            Edges are oriented such that orientation index 0 is in {F, D, B, T}. 
            We want the Left face's color thus this face's correct index to display is (e_ori + 1) % 2 """
        row = 4 * n + 2
        e = 10
        for i in range(s):
            e_idx = e * s + i
            e_ori = int(e_orientations[e_idx])
            up, left = e_ori % 2, (e_ori + 1) % 2
            img[row, 2+i] = self.edge_c_map[edges[e_idx], up]  # Up face's edge
            img[row + 1, 2+i] = self.edge_c_map[edges[e_idx], left]
        e = 7
        for i in range(s):
            e_idx = e * s + (s - 1 - i)  # This edge should be displayed reversed
            e_ori = int(e_orientations[e_idx])
            back, left = e_ori % 2, (e_ori + 1) % 2
            img[row+2+i, 0] = self.edge_c_map[edges[e_idx], back]  # Back face's edge
            img[row+2+i, 1] = self.edge_c_map[edges[e_idx], left]
        e = 1
        for i in range(s):
            e_idx = e * s + i
            e_ori = int(e_orientations[e_idx]) + 1  # Orientation is inverted since edge is viewed inverted
            left, front = e_ori % 2, (e_ori + 1) % 2
            img[row+2+i, -1] = self.edge_c_map[edges[e_idx], front]  # Front face's edge
            img[row+2+i, -2] = self.edge_c_map[edges[e_idx], left]
        e = 4
        for i in range(s):
            e_idx = e * s + (s - 1 - i)  # This edge should be displayed reversed
            e_ori = int(e_orientations[e_idx]) + 1  # Orientation is inverted since edge is viewed inverted
            front, down = e_ori % 2, (e_ori + 1) % 2
            img[row+2+s, 2+i] = self.edge_c_map[edges[e_idx], front]
            img[row+2+s + 1, 2+i] = self.edge_c_map[edges[e_idx], down]  # Down face's edge

        """ Do Right side's edges:
            Edges are oriented such that orientation index 0 is in {F, D, B, T}. 
            We want the Right face's color thus this face's correct index to display is (e_ori + 1) % 2 """
        row = 5 * n + 4
        e = 11
        for i in range(s):
            e_idx = e * s + (s - 1 - i)  # This edge should be displayed reversed
            e_ori = int(e_orientations[e_idx])
            up, right = e_ori % 2, (e_ori + 1) % 2
            img[row, 2 + i] = self.edge_c_map[edges[e_idx], up]  # Up face's edge
            img[row + 1, 2 + i] = self.edge_c_map[edges[e_idx], right]
        e = 2
        for i in range(s):
            e_idx = e * s + i
            e_ori = int(e_orientations[e_idx]) + 1
            right, front = e_ori % 2, (e_ori + 1) % 2
            img[row + 2 + i, 0] = self.edge_c_map[edges[e_idx], front]  # Front face's edge
            img[row + 2 + i, 1] = self.edge_c_map[edges[e_idx], right]
        e = 8
        for i in range(s):
            e_idx = e * s + (s - 1 - i)  # This edge should be displayed reversed
            e_ori = int(e_orientations[e_idx]) + 1
            right, back = e_ori % 2, (e_ori + 1) % 2
            img[row + 2 + i, -1] = self.edge_c_map[edges[e_idx], back]  # Back face's edge
            img[row + 2 + i, -2] = self.edge_c_map[edges[e_idx], right]
        e = 5
        for i in range(s):
            e_idx = e * s + i
            e_ori = int(e_orientations[e_idx])
            down, right = e_ori % 2, (e_ori + 1) % 2  # Orientation is inverted since edge is viewed upside down
            img[row + 2 + s, 2 + i] = self.edge_c_map[edges[e_idx], right]
            img[row + 2 + s + 1, 2 + i] = self.edge_c_map[edges[e_idx], down]  # Down face's edge

        # ######## Faces ###################################################
        """ Do Front->Down->Back->Up side's faces:
            Face pieces are ordered right-to-left, top-to-bottom. """
        for f in range(4):
            for i in range(s):
                row = f * n + i + 2
                for j in range(s):
                    img[row, 2 + j] = self.face_c_map[faces[f*a + i*s + j]]
        """ Do Left side's faces: """
        start_row = 4 * n + 2
        f = 4
        for i in range(s):
            row = start_row + i + 2
            for j in range(s):
                img[row, 2 + j] = self.face_c_map[faces[f * a + i * s + j]]
        """ Do Right side's faces: """
        start_row = 5 * n + 2 + 2
        f = 5
        for i in range(s):
            row = start_row + i + 2
            for j in range(s):
                img[row, 2 + j] = self.face_c_map[faces[f * a + i * s + j]]

        img_resized = cv2.resize(img, (img.shape[1]*18, img.shape[0]*18), interpolation=cv2.INTER_AREA)
        cv2.imshow("image", img_resized[:,:,::-1])
        cv2.waitKey()

    def visualize_compact(self, faces, edges, e_orientations, corners, c_orientations):
        img = np.zeros((self.n * 6, self.n, 3))

        n = self.n
        s = n - 2
        a = s ** 2
        # ######## Corners ###################################################
        """ Do Front->Down->Back->Up side's corners:
            Corners are oriented such that side on position 0 is in {F, D, B, or T}, and go clockwise from there. """
        for c in range(8):
            row = c // 2 * n
            c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
            if c % 2 == 0:
                # If corner is on top-left, corner face orientation will be: - position 0 displayed in [F, D, B, T]
                #                                                            - position 2 displayed in [T, F, D, B].
                front, top = c_ori, (c_ori + 2) % 3
                img[(row-1)%(n*4), 0] = self.corner_c_map[corners[c], top]
                img[row, 0] = self.corner_c_map[corners[c], front]
            else:
                # If corner is on top-right, corner face orientation will be: - position 0 displayed in [F, D, B, T]
                #                                                             - position 1 displayed in [T, F, D, B].
                front, top = c_ori, (c_ori + 1) % 3
                img[(row-1)%(n*4), -1] = self.corner_c_map[corners[c], top]
                img[row, -1] = self.corner_c_map[corners[c], front]

        """ Do Left side's corners:
            Corner's are centered on {F, D, B, or T}, and corner faces are indexed clockwise. 
            Thus Left face is always on index 1 of top-left corners. """
        row = 4 * n
        c = 6  # Corner 6 should be top left
        c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
        img[row, 0] = self.corner_c_map[corners[c], (c_ori + 1)%3]
        c = 0  # Corner 0 should be bottom left
        c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
        img[row, -1] = self.corner_c_map[corners[c], (c_ori + 1)%3]

        row = 5 * n - 1
        c = 4  # Corner 4 should be bottom left
        c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
        img[row, 0] = self.corner_c_map[corners[c], (c_ori + 1)%3]
        c = 2  # Corner 2 should be bottom right
        c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
        img[row, -1] = self.corner_c_map[corners[c], (c_ori + 1)%3]

        """ Do Right side's corners:
            Corner's are centered on {F, D, B, or T}, and corner faces are indexed clockwise. 
            Thus Right face is always on index 2 of top-right corners. """
        row = 5 * n
        c = 1
        c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
        img[row, 0] = self.corner_c_map[corners[c], (c_ori + 2)%3]
        c = 7
        c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
        img[row, -1] = self.corner_c_map[corners[c], (c_ori + 2)%3]

        row = 6 * n - 1
        c = 3
        c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
        img[row, 0] = self.corner_c_map[corners[c], (c_ori + 2)%3]
        c = 5
        c_ori = ba2int(c_orientations[c * 2:(c + 1) * 2])
        img[row, -1] = self.corner_c_map[corners[c], (c_ori + 2)%3]

        # ######## Edges ###################################################
        """ Do Front->Down->Back->Up side's edges:
            Edges are oriented such that for each face in {F, D, B, T}, the upper, left, and right edges
            belong to that face and thus their side of the edge will be on index 0. """
        for e in range(12):  # Cube face order ir [F, D, B, T]
            row = e // 3 * n
            for i in range(s):
                e_idx = e * s + i
                e_ori = int(e_orientations[e_idx])
                if e % 3 == 0:  # Do this cubeface's upper edge
                    img[(row-1)%(n*4), 1 + i] = self.edge_c_map[edges[e_idx], (e_ori + 1) % 2]
                    img[row, 1 + i] = self.edge_c_map[edges[e_idx], e_ori]
                elif e % 3 == 1:  # Do this cubeface's left edge
                    img[row + 1 + i, 0] = self.edge_c_map[edges[e_idx], e_ori]
                else:  # Do this cubeface's right edge
                    img[row + 1 + i, -1] = self.edge_c_map[edges[e_idx], e_ori]

        """ Do Left side's (green) edges:
            Edges are oriented such that orientation index 0 is in {F, D, B, T}. 
            We want the side face's color thus: correct color to display is (e_ori + 1) % 2 """
        row = 4 * n
        e = 10
        for i in range(s):
            e_idx = e * s + i
            e_ori = int(e_orientations[e_idx])
            img[row, 1 + i] = self.edge_c_map[edges[e_idx], (e_ori + 1) % 2]
        e = 7
        for i in range(s):
            e_idx = e * s + (s - 1 - i)  # This edge should be displayed reversed order
            e_ori = int(e_orientations[e_idx])
            img[row + 1 + i, 0] = self.edge_c_map[edges[e_idx], (e_ori + 1) % 2]
        e = 1
        for i in range(s):
            e_idx = e * s + i
            e_ori = int(e_orientations[e_idx])
            img[row + 1 + i, -1] = self.edge_c_map[edges[e_idx], (e_ori + 1) % 2]
        e = 4
        for i in range(s):
            e_idx = e * s + (s - 1 - i)  # This edge should be displayed reversed order
            e_ori = int(e_orientations[e_idx])
            img[row + 1 + s, 1 + i] = self.edge_c_map[edges[e_idx], (e_ori + 1) % 2]

        """ Do Right side's (blue) edges:
            Edges are oriented such that orientation index 0 is in {F, D, B, T}. 
            We want the Left face's color thus this face's correct index to display is (e_ori + 1) % 2 """
        row = 5 * n
        e = 11
        for i in range(s):
            e_idx = e * s + (s - 1 - i)  # This edge should be displayed reversed order
            e_ori = int(e_orientations[e_idx])
            img[row, 1 + i] = self.edge_c_map[edges[e_idx], (e_ori + 1) % 2]
        e = 2
        for i in range(s):
            e_idx = e * s + i
            e_ori = int(e_orientations[e_idx])
            img[row + 1 + i, 0] = self.edge_c_map[edges[e_idx], (e_ori + 1) % 2]
        e = 8
        for i in range(s):
            e_idx = e * s + (s - 1 - i)  # This edge should be displayed reversed order
            e_ori = int(e_orientations[e_idx])
            img[row + 1 + i, -1] = self.edge_c_map[edges[e_idx], (e_ori + 1) % 2]
        e = 5
        for i in range(s):
            e_idx = e * s + i
            e_ori = int(e_orientations[e_idx])
            img[row + 1 + s, 1 + i] = self.edge_c_map[edges[e_idx], (e_ori + 1) % 2]

        # ######## Faces ###################################################
        """ Do Front->Down->Back->Up side's faces:
            Face pieces are ordered right-to-left, top-to-bottom. """
        for f in range(4):  # Cube face order ir [F, D, B, T]
            for i in range(s):
                row = 1 + f * n + i
                for j in range(s):
                    img[row, 1 + j] = self.face_c_map[faces[f * a + i * s + j]]
        """ Do Left side's faces: """
        start_row = 4 * n
        f = 4
        for i in range(s):
            row = 1 + start_row + i
            for j in range(s):
                img[row, 1 + j] = self.face_c_map[faces[f * a + i * s + j]]
        """ Do Right side's faces: """
        start_row = 5 * n
        f = 5
        for i in range(s):
            row = 1 + start_row + i
            for j in range(s):
                img[row, 1 + j] = self.face_c_map[faces[f * a + i * s + j]]

        img_resized = cv2.resize(img, (img.shape[1] * 18, img.shape[0] * 18), interpolation=cv2.INTER_AREA)
        cv2.imshow("image", img_resized[:, :, ::-1])
        cv2.waitKey()

    def get_side_2d(self, side: str):
        """ Get 2d representation of a side
        :param side: F """
        if isinstance(side, str):
            side = self.SIDE_ALIAS[side]


if __name__ == "__main__":
    c = Cube(7)
    cviz = CubeVisualizer(7)
    cviz.visualize(c.face_positions, c.edge_positions, c.edge_orientations, c.corner_positions, c.corner_orientations)


import unittest
import numpy as np
from xms.geometry.winding import winding_numbers,inside

CUBE_V=np.array([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)],float)
CUBE_T=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])


class Winding(unittest.TestCase):
    def test_closed_cube_inside_outside_any_orientation(self):
        points=[[.5,.5,.5],[.1,.9,.2],[1.5,.5,.5],[-.2,-.2,-.2]]
        np.testing.assert_allclose(np.abs(winding_numbers(points,CUBE_V,CUBE_T)),[1,1,0,0],atol=1e-9)
        np.testing.assert_array_equal(inside(points,CUBE_V,CUBE_T[:,::-1]),[True,True,False,False])
    def test_open_cube_degrades_smoothly(self):
        # Remove the top face (z=1): deep interior points stay inside, points near the hole drop toward 0.5.
        open_t=np.array([t for t in CUBE_T if not all(CUBE_V[t][:,2]==1)])
        w=np.abs(winding_numbers([[.5,.5,.1],[.5,.5,.95],[.5,.5,1.5]],CUBE_V,open_t))
        self.assertGreater(w[0],.8);self.assertLess(w[1],w[0]);self.assertLess(w[2],.5)


if __name__=='__main__':unittest.main()

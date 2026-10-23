"""Real visual-test failure: colour camouflage erased a measured 12 cm curb."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module
GB=load_package_module('ground_bands')

class HeightSegmentation(unittest.TestCase):
    def profile(self,start=38,end=42,height=.12):
        n=64
        return dict(lo=0,bin=.1,n=np.full(n,100),lab=np.tile([60.,8.,20.],(n,1)),
                    tex=np.ones(n),z=np.r_[np.zeros(start),np.full(end-start,height),np.zeros(n-end)])
    def stats(self,p):
        return GB._merge(p,GB._segments(p)[0])
    def test_camouflaged_raised_band_survives(self):
        st=self.stats(self.profile())
        self.assertEqual([(x['k0'],x['k1']) for x in st],[(0,38),(38,42),(42,64)])
        self.assertAlmostEqual(st[1]['z'],.12)
    def test_two_bin_raised_band_not_absorbed_as_noise(self):
        st=self.stats(self.profile(end=40))
        self.assertEqual([(x['k0'],x['k1']) for x in st],[(0,38),(38,40),(40,64)])
    def test_matching_colour_does_not_prevent_curb_class(self):
        st=self.stats(self.profile())
        bands=[dict(s0=x['k0']*.1,s1=x['k1']*.1,width=(x['k1']-x['k0'])*.1,
                    lab=x['lab'],tex=x['tex'],z=x['z']) for x in st]
        GB._classify(bands,1.,{'has_road':True})
        self.assertEqual([b['kind'] for b in bands],['main','curb','shoulder'])
    def test_flat_colours_do_not_create_height_edges(self):
        self.assertEqual(len(self.stats(self.profile(height=0))),1)
    def test_gentle_slope_does_not_create_step_edges(self):
        p=self.profile(height=0);p['z']=np.arange(64)*.01
        self.assertEqual(len(self.stats(p)),1)
    def test_short_missing_bin_preserves_height_edge(self):
        p=self.profile();p['lab'][37]=np.nan;p['z'][37]=np.nan;p['n'][37]=0
        st=self.stats(p)
        self.assertTrue(any(x['k0']==38 and x['k1']==42 for x in st))

if __name__=='__main__':unittest.main()

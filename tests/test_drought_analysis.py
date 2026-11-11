import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "drought"))

from run_drought_analysis import (  # noqa: E402
    first_attainment,
    windows,
    reduce_period,
    weighted_median,
    block_ci,
)

class AnalysisTests(unittest.TestCase):
    def test_recovery_and_missing(self):
        a=np.array([[80,95,101],[np.nan,101,102],[80,81,82],[np.nan,np.nan,np.nan]])
        t=first_attainment(a,np.full(4,100.),1.,[2003,2004,2005],2002)
        np.testing.assert_allclose(t,[3,2,np.nan,np.nan],equal_nan=True)
    def test_two_consecutive(self):
        a=np.array([[101,90,103],[101,102,90],[90,101,102]])
        t=first_attainment(a,np.full(3,100.),1.,[2003,2004,2005],2002,True)
        np.testing.assert_allclose(t,[np.nan,1,2],equal_nan=True)
    def test_empty_followup(self):
        t=first_attainment(np.empty((2,0)),np.array([100,100]),.9,[],2002)
        self.assertTrue(np.isnan(t).all())
    def test_new_event_censors(self):
        events=[{'start':2000,'end':2000,'years':[2000]}, {'start':2002,'end':2002,'years':[2002]}]
        pre,post,cross,reason=windows(events[0],events,{2000,2002},'AGB',{}, {})
        self.assertEqual(post,[2001]); self.assertEqual(reason,'new_dry_episode')
    def test_sensor_censors(self):
        e={'start':2002,'end':2002,'years':[2002]}
        pre,post,_,reason=windows(e,[e],{2002},'AGB',{}, {'same_sensor':True})
        self.assertEqual(post,[]); self.assertEqual(reason,'sensor_transition')
    def test_insufficient_baseline(self):
        p=pd.DataFrame({2013:[np.nan,2.],2014:[1.,3.]})
        b=reduce_period(p,[2013,2014],2)
        self.assertTrue(np.isnan(b.iloc[0])); self.assertEqual(b.iloc[1],2.5)
    def test_weighted_median(self):
        self.assertEqual(weighted_median([1,2,3],[1,8,1]),2.)
    def test_cluster_interval(self):
        p=pd.DataFrame({'value':[1.,1.,1.,1.,1.], 'area_km2':[1]*5,'block_id':list('abcde')})
        m,l,h,n=block_ci(p,'value')
        self.assertEqual((m,l,h,n),(1.,1.,1.,5))

if __name__=='__main__': unittest.main(verbosity=2)

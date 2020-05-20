from unittest import TestCase

import numpy as np
import os
import shutil
import warnings

from .. import basis_generate as ddbg
from ..process_data import full_fit

from . import __file__

folder = os.path.dirname(__file__)
warnings.filterwarnings('ignore', 'More than 20 figures have been opened.',
                        RuntimeWarning)


class TestImage(TestCase):
    def test_UV_bg(self):
        settingsfn = os.path.join(folder,
                                  'test_data/settings.json')
        metadatafn = os.path.join(folder,
                                  'test_data/UVim300ulph_Metadata.json')
        tempdir = "Test_tempdir"
        infos = full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], 2.5e-9)
        self.assertLess(infos['Radius'], 5e-9)

    def test_scans(self):
        settingsfn = os.path.join(folder,
                                  'test_data/settings.json')
        metadatafn = os.path.join(folder,
                                  'test_data/scans/metadata.json')
        tempdir = "Test_tempdir"
        infos = full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], 2e-9)
        self.assertLess(infos['Radius'], 3e-9)

    def test_single_scans(self):
        settingsfn = os.path.join(folder,
                                  'test_data/exp1_z2_x2_settings.json')
        metadatafn = os.path.join(folder,
                                  'test_data/exp1_z2_x2_metadata.json')
        tempdir = "Test_tempdir"
        infos = full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], 3e-9)
        self.assertLess(infos['Radius'], 4e-9)

    def test_UV_bg_poly_2(self):
        settingsfn = os.path.join(
            folder, 'test_data/settings_poly_2.json')
        metadatafn = os.path.join(
            folder, 'test_data/UVim300ulph_Metadata.json')
        tempdir = "Test_tempdir"
        try:
            infos = full_fit(settingsfn, metadatafn, tempdir)
        except RuntimeError as e:
            if e.args[0] != "Monodisperse":
                raise
            return

        shutil.rmtree(tempdir)
        Rs, spectrum = infos['Radius']
        self.assertGreater(Rs[np.argmax(spectrum)], 2.5e-9)
        self.assertLess(Rs[np.argmax(spectrum)], 5e-9)

    def test_bright(self):
        settingsfn = os.path.join(
            folder, 'test_data/settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/Brightim900ulph_Metadata.json')
        tempdir = "Test_tempdir"
        infos = full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], .2e-9)
        self.assertLess(infos['Radius'], 1e-9)

    def test_film(self):
        settingsfn = os.path.join(
            folder, 'test_data/settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/327.68ul-h-50um device_Metadata.json')
        tempdir = "Test_tempdir"
        infos = full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(np.sum(np.isfinite(infos.loc[:, 'Radius'])),
                           np.sum(np.isnan(infos.loc[:, 'Radius'])))
        self.assertFalse(np.max(infos.loc[:, 'Radius']) > 4e-9)

    def test_12pos(self):
        settingsfn = os.path.join(
            folder, 'test_data/settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/350ulh_12pos/metadata.json')
        tempdir = "Test_tempdir"
        infos = \
            full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], 2.5e-9)
        self.assertLess(infos['Radius'], 4.5e-9)

    def test_bright_walls(self):
        settingsfn = os.path.join(
            folder, 'test_data/settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/1tf_001tweentico75_100ulhr_Metadata.json')
        tempdir = "Test_tempdir"
        infos = \
            full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], 2e-9)
        self.assertLess(infos['Radius'], 4e-9)

    def test_scans_confocal_10pM(self):
        settingsfn = os.path.join(
            folder, 'test_data/settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/10pM_RNAseA488_20mM_Pi_200uM_EDTA_pH8_0p1%_Tween_400uW_-100ulh_200_steps_10s_9_steps_metadata.json')
        tempdir = "Test_tempdir"
        infos = \
            full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], 1e-9)
        self.assertLess(infos['Radius'], 3e-9)

    def test_scans_confocal_100pM(self):
        settingsfn = os.path.join(
            folder, 'test_data/settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/100pM_RNAseA488_20mM_Pi_200uM_EDTA_pH8_0p1%_Tween_400uW_-100ulh_200_steps_5s_5_steps_metadata.json')
        tempdir = "Test_tempdir"
        infos = \
            full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], 1e-9)
        self.assertLess(infos['Radius'], 3e-9)

    def test_will(self):
        settingsfn = os.path.join(
            folder, 'test_data/will_stg/settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/will_stg/metadata.json')
        tempdir = "Test_tempdir"
        infos = \
            full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(np.min(infos['Radius']), 1e-9)
        self.assertLess(np.max(infos['Radius']), 3e-9)

    def test_Slice(self):
        settingsfn = os.path.join(
            folder, 'test_data/settings_slice.json')
        metadatafn = os.path.join(
            folder, 'test_data/1tf_001tweentico75_100ulhr_Metadata.json')
        tempdir = "Test_tempdir"
        infos = \
            full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], 2e-9)
        self.assertLess(infos['Radius'], 4e-9)

    def test_opticsbg(self):
        settingsfn = os.path.join(folder,
                                  'test_data/settings.json')
        metadatafn = os.path.join(folder,
                                  'test_data/M4-5_metadata.json')
        tempdir = "Test_tempdir"
        infos = \
            full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'], 4e-9)
        self.assertLess(infos['Radius'], 4.6e-9)

    def test_electrophoresis(self):
        basisfn = os.path.join(folder,
                               'test_data/q12_basis1_002.dat')
        initfn = os.path.join(folder,
                              'test_data/q12_init.dat')

        qelectron = 1.6021766e-19

        viscosity = 1e-3
        temperature = 295
        Wz = 50e-6
        Wy = 1400e-6
        D = 1.2*1e-10
        Q = 308
        readingpos = 5.6e-3
        V = 2
        Zgrid = 7
        q = 12*qelectron

        kT = 1.38e-23*temperature

        E = V / Wy
        r = kT / (6 * np.pi * viscosity * D)

        prof2 = np.loadtxt(basisfn)
        init = np.loadtxt(initfn)
        init /= init.sum()
        init *= prof2.sum()

        prof = ddbg.getprofiles(init, Q, [r], [readingpos],
                                Wy, Wz,
                                viscosity=viscosity, temperature=temperature,
                                Zgrid=Zgrid,
                                muEoD=q*E/kT)
        prof = np.squeeze(prof)

        self.assertLess(np.max(np.abs(
            (prof-prof2)/np.max(prof[:-3]))[:-2]), .1)

    def test_scan(self):
        settingsfn = os.path.join(
            folder, 'test_data/martin/martin_settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/martin/metadata.json')
        tempdir = "Test_tempdir"
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    "Missing pixels, scan not large enough",
                                    RuntimeWarning)
            infos = full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(np.min(infos['Radius']), 1e-9)
        self.assertLess(np.max(infos['Radius']), 3e-9)

    def test_mobility(self):
        settingsfn = os.path.join(
            folder, 'test_data/tom_mobility_r/settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/tom_mobility_r/stack_metadata.json')
        tempdir = "Test_tempdir"
        infos = full_fit(settingsfn, metadatafn, tempdir)
        shutil.rmtree(tempdir)
        self.assertGreater(infos['Radius'][0], 2.5e-9)
        self.assertLess(infos['Radius'][0], 4e-9)

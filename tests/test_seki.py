from bamboo import test_seki as ctest


class TestSeki():

    @classmethod
    def setup_class(self):
        ctest.setup()
        
    @classmethod
    def teardown_class(self):
        pass
    
    def test_seki_0(self):
        ctest.test_seki_0()

    def test_seki_1(self):
        ctest.test_seki_1()

    def test_seki_2(self):
        ctest.test_seki_2()

    def test_seki_3(self):
        ctest.test_seki_3()

    def test_seki_4(self):
        ctest.test_seki_4()

    def test_seki_5(self):
        ctest.test_seki_5()

    def test_seki_6(self):
        ctest.test_seki_6()

    def test_seki_7(self):
        ctest.test_seki_7()

    def test_seki_8(self):
        ctest.test_seki_8()

    def test_seki_9(self):
        ctest.test_seki_9()

    def test_bent4(self):
        ctest.test_bent4()

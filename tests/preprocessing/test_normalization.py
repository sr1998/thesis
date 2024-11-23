import pandas as pd
from preprocessing.normalization import total_sum_scaling

class TestTotalSumScaling():

    def test_non_square_matrix(self):
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        result = total_sum_scaling(data)
        expected = pd.DataFrame({
            'A': [1/5, 2/7, 3/9],
            'B': [4/5, 5/7, 6/9]
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        data = pd.DataFrame()
        result = total_sum_scaling(data)
        expected = pd.DataFrame()
        pd.testing.assert_frame_equal(result, expected)

    def test_single_value(self):
        data = pd.DataFrame({'A': [1.0]})
        result = total_sum_scaling(data)
        expected = pd.DataFrame({'A': [1.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_zero_values(self):
        data = pd.DataFrame({
            'A': [0, 0, 0],
            'B': [0, 0, 0],
            'C': [0, 0, 0]
        })
        result = total_sum_scaling(data)
        expected = pd.DataFrame({
            'A': [0, 0, 0],
            'B': [0, 0, 0],
            'C': [0, 0, 0]
        })
        print(result)
        pd.testing.assert_frame_equal(result, expected)

    def test_negative_values(self):
        data = pd.DataFrame({
            'A': [-1, -2, 1],
            'B': [-4, -5, 6],
            'C': [7, -8, -9]
        })
        result = total_sum_scaling(data)
        expected = pd.DataFrame({
            'A': [-1/2, 2/15, -1/2],
            'B': [-4/2, 5/15, -6/2],
            'C': [7/2, 8/15, 9/2]
        })
        pd.testing.assert_frame_equal(result, expected)

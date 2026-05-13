import unittest
import torch
import sys
import os
import importlib.util

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestTCNGRUArchitecture(unittest.TestCase):
    def setUp(self):
        # Programmatic importation to handle names starting with digits
        spec = importlib.util.spec_from_file_location(
            "tcn_gru_module",
            os.path.join(os.path.dirname(__file__), "../src/02_TCN_GRU_Architecture_Training.py")
        )
        self.module = importlib.util.module_from_spec(spec)
        # Prevent execution of the top-level loading script from throwing if files missing
        # Though we added if __name__ == '__main__': blocks for some of it, it might still have data loading.
        # Actually data loading is outside if __name__ == '__main__' block: `with open(...) as f`
        # Let's bypass the sys.modules mapping if possible or just import.
        # Instead, just mock or let it run, it has a try-except block for data loading that catches FileNotFoundError!
        spec.loader.exec_module(self.module)
        self.TCNGRU = getattr(self.module, "TCNGRU")

    def test_model_architecture(self):
        """Test the model architecture dimensions and forward pass."""
        # Test with GEFCom data dimensions
        batch_size = 32
        input_size = 31  # 24 history + 7 features
        output_size = 6   # 6-step forecast

        model = self.TCNGRU(
            input_size=input_size,
            tcn_channels=[32, 64, 32],
            gru_hidden_size=64,
            output_size=output_size,
            tcn_kernel_size=3,
            tcn_dropout=0.1,
            gru_dropout=0.1
        )

        # Test forward pass
        x = torch.randn(batch_size, input_size)
        output = model(x)

        # Verify output shape
        self.assertEqual(output.shape, (batch_size, output_size))

if __name__ == "__main__":
    unittest.main()

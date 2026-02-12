import importlib.util
import pathlib
import types
import unittest
from unittest.mock import Mock, patch

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "vcoder" / "data" / "websight.py"
spec = importlib.util.spec_from_file_location("websight_module", MODULE_PATH)
websight_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(websight_module)
load_websight_dataset = websight_module.load_websight_dataset


class FakeDataset:
    def __init__(self, size: int = 10):
        self.size = size
        self.selected = None
        self.taken = None
        self.processor = None

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter({"value": i} for i in range(self.size))

    def select(self, idxs):
        self.selected = list(idxs)
        self.size = len(self.selected)
        return self

    def take(self, n):
        self.taken = n
        return iter({"value": i} for i in range(n))

    def map(self, processor):
        self.processor = processor
        return self


class LoadWebSightDatasetTests(unittest.TestCase):
    def _run_with_fake_datasets(self, fake_dataset, **kwargs):
        mocked_loader = Mock(return_value=fake_dataset)
        dataset_factory = types.SimpleNamespace(from_list=Mock(return_value=FakeDataset()))
        fake_datasets_module = types.SimpleNamespace(load_dataset=mocked_loader, Dataset=dataset_factory)

        with patch.dict("sys.modules", {"datasets": fake_datasets_module}):
            output = load_websight_dataset(lambda row: row, **kwargs)

        return output, mocked_loader

    def test_passes_split_to_load_dataset(self):
        fake_dataset = FakeDataset()
        output, mocked_loader = self._run_with_fake_datasets(fake_dataset, split="train[:1%]")

        mocked_loader.assert_called_once()
        self.assertEqual(mocked_loader.call_args.kwargs["split"], "train[:1%]")
        self.assertIsNotNone(output.processor)

    def test_max_samples_selects_for_non_streaming(self):
        fake_dataset = FakeDataset(size=20)
        self._run_with_fake_datasets(fake_dataset, max_samples=5)
        self.assertEqual(fake_dataset.selected, [0, 1, 2, 3, 4])

    def test_max_samples_take_for_streaming(self):
        fake_dataset = FakeDataset(size=20)
        output, mocked_loader = self._run_with_fake_datasets(fake_dataset, streaming=True, max_samples=5)
        self.assertEqual(mocked_loader.call_args.kwargs["streaming"], True)
        self.assertIsNotNone(output.processor)

    def test_streaming_requires_max_samples(self):
        fake_dataset = FakeDataset(size=20)
        with self.assertRaises(ValueError):
            self._run_with_fake_datasets(fake_dataset, streaming=True)

    def test_rejects_non_positive_max_samples(self):
        fake_dataset = FakeDataset()
        with self.assertRaises(ValueError):
            self._run_with_fake_datasets(fake_dataset, max_samples=0)


if __name__ == "__main__":
    unittest.main()

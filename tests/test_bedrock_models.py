"""
Catalog consistency checks for cmn/bedrock_models.py.

Run from the project root:  python -m unittest discover -s tests -v
"""
import dataclasses
import unittest

from cmn.bedrock_models import (
    ALL_FEATURES, CLAUDE_NO_SAMPLING_64K, FOUNDATION_MODELS, Feature, FoundationModel,
    FoundationModelCatalog, InferenceParameterSet, Provider, all_features_except,
)

PAGE_DEFAULT_MODEL = "global.anthropic.claude-sonnet-5"  # pages/3_5_13_converse_demo.py


class CatalogTest(unittest.TestCase):

    def test_every_model_is_complete(self):
        for model in FOUNDATION_MODELS:
            with self.subTest(model.model_id):
                self.assertTrue(model.name.strip())
                self.assertIsInstance(model.provider, Provider)
                self.assertIsInstance(model.inference_parameters, InferenceParameterSet)
                self.assertLessEqual(model.features, ALL_FEATURES)
                self.assertNotIn(" ", model.model_id, "model IDs never contain spaces")

    def test_max_tokens_range_is_ordered(self):
        for model in FOUNDATION_MODELS:
            max_tokens = model.inference_parameters.max_tokens
            with self.subTest(model.model_id):
                self.assertLessEqual(max_tokens.min_value, max_tokens.default_value)
                self.assertLessEqual(max_tokens.default_value, max_tokens.max_value)

    def test_listed_display_names_are_unique(self):
        names = [model.display_name for model in FOUNDATION_MODELS.listed_models()]
        self.assertEqual(len(names), len(set(names)))

    def test_page_default_is_listed(self):
        self.assertIn(FOUNDATION_MODELS.find(PAGE_DEFAULT_MODEL), FOUNDATION_MODELS.listed_models())

    def test_listed_claude_models_use_global_profiles(self):
        for model in FOUNDATION_MODELS.listed_models():
            if model.provider is Provider.ANTHROPIC:
                self.assertTrue(model.model_id.startswith("global."), model.model_id)

    def test_find(self):
        model = FOUNDATION_MODELS.find("global.anthropic.claude-opus-5-5")
        self.assertEqual(model.model_id, "global.anthropic.claude-opus-5-5")
        self.assertEqual(model.provider, "Anthropic")
        self.assertIsNone(FOUNDATION_MODELS.find("no.such-model"))

    def test_duplicate_model_ids_are_rejected(self):
        model = FOUNDATION_MODELS.find("global.anthropic.claude-opus-5-5")
        with self.assertRaises(ValueError):
            FoundationModelCatalog([model, model])


class FoundationModelTest(unittest.TestCase):

    def test_is_immutable(self):
        model = FOUNDATION_MODELS.find("global.anthropic.claude-opus-5-5")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            model.name = "changed"

    def test_claude_5_sends_no_sampling_parameters(self):
        for model_id in ("global.anthropic.claude-opus-5-5", "global.anthropic.claude-sonnet-5-5",
                         "global.anthropic.claude-haiku-5-5", "global.anthropic.claude-fable-5-1"):
            parameters = FOUNDATION_MODELS.find(model_id).inference_parameters
            with self.subTest(model_id):
                self.assertFalse(parameters.temperature.is_supported)
                self.assertFalse(parameters.top_p.is_supported)
                self.assertFalse(parameters.top_k.is_supported)

    def test_is_feature_supported(self):
        claude = FOUNDATION_MODELS.find("global.anthropic.claude-opus-5-5")
        cohere = FOUNDATION_MODELS.find("cohere.command-r-v1:0")
        self.assertTrue(claude.is_feature_supported(Feature.VISION))
        self.assertTrue(claude.is_feature_supported("vision"), "plain strings work too")
        self.assertFalse(cohere.is_feature_supported(Feature.VISION))
        self.assertFalse(cohere.is_feature_supported("no_such_feature"))

    def test_region(self):
        self.assertEqual(FOUNDATION_MODELS.find("us.writer.palmyra-x5-v1:0").region, "us-west-2")
        self.assertIsNone(FOUNDATION_MODELS.find("global.anthropic.claude-opus-5-5").region)

    def test_inference_parameter_sets_are_shared(self):
        model = FOUNDATION_MODELS.find("global.anthropic.claude-opus-5-5")
        self.assertIs(model.inference_parameters, CLAUDE_NO_SAMPLING_64K)


class DisplayNameTest(unittest.TestCase):

    def display_name(self, model_id):
        return FOUNDATION_MODELS.find(model_id).display_name

    def test_global_has_no_suffix(self):
        self.assertEqual(self.display_name("global.anthropic.claude-opus-5-5"), "Claude Opus 5.5")

    def test_us_suffix(self):
        self.assertEqual(self.display_name("us.amazon.nova-premier-v1:0"), "Amazon Nova Premier (US)")

    def test_on_demand_id_has_no_suffix(self):
        self.assertEqual(self.display_name("amazon.nova-pro-v1:0"), "Amazon Nova Pro")


class EnumTest(unittest.TestCase):

    def test_provider_prints_as_its_value(self):
        self.assertEqual(f"{Provider.ANTHROPIC}", "Anthropic")
        self.assertEqual(str(Provider.ANTHROPIC), "Anthropic")

    def test_all_features_except_rejects_unknown_feature(self):
        with self.assertRaises(ValueError):
            all_features_except("visoin")


if __name__ == "__main__":
    unittest.main()

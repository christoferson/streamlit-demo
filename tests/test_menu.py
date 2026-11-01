import re
import unittest
from pathlib import Path

from cmn.view.menu import MENU_SECTIONS

ROOT = Path(__file__).resolve().parent.parent
ALL_PAGES = [page for pages in MENU_SECTIONS.values() for page in pages]


class MenuTest(unittest.TestCase):

    def test_every_page_file_exists(self):
        for page in ALL_PAGES:
            self.assertTrue((ROOT / page.path).is_file(), page.path)

    def test_pages_and_url_paths_are_unique(self):
        paths = [page.path for page in ALL_PAGES]
        # Streamlit's default url_path: the file name minus its first number
        url_paths = [re.sub(r"^[0-9]*[_ -]*", "", Path(path).stem) for path in paths]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(url_paths), len(set(url_paths)))

    def test_icons_are_material_and_summaries_present(self):
        for page in ALL_PAGES:
            self.assertRegex(page.icon, r"^:material/[a-z_]+:$", page.path)
            self.assertTrue(page.summary.strip(), page.path)

    def test_every_page_has_details_and_something_to_try(self):
        for page in ALL_PAGES:
            self.assertTrue(page.details, page.path)
            self.assertTrue(page.try_this.strip(), page.path)

    def test_text_has_no_markdown_brackets_that_break_badges(self):
        for page in ALL_PAGES:
            for service in page.services:
                self.assertNotRegex(service, r"[\[\]]", page.path)

    def test_featured_pages_fit_the_home_page_cards(self):
        featured = [page for page in ALL_PAGES if page.featured]
        self.assertTrue(1 <= len(featured) <= 6)

    def test_menu_md_lists_every_page(self):
        menu_md = (ROOT / "MENU.md").read_text(encoding="utf-8")
        for page in ALL_PAGES:
            self.assertIn(Path(page.path).name, menu_md)


if __name__ == "__main__":
    unittest.main()

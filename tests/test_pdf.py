from PIL import Image, ImageDraw

from rag.pdf import PageImage, is_blank, page_filename, page_image_path, render_pages
from tests.fakes import make_pdf


def test_is_blank_detects_white_page():
    assert is_blank(Image.new("RGB", (100, 100), "white"))


def test_is_blank_false_for_content():
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 60, 60), fill="black")
    assert not is_blank(image)


def test_is_blank_tolerates_scan_specks():
    image = Image.new("RGB", (1000, 1000), "white")
    for x in range(20):
        image.putpixel((x * 7, 500), (0, 0, 0))
    assert is_blank(image)


def test_render_skips_blank_pages(tmp_path):
    pdf = tmp_path / "m.pdf"
    make_pdf(pdf, [False, True])
    out = tmp_path / "out"
    assert render_pages(pdf, out, dpi=72) == [PageImage(pdf_page=2, path=out / "p02.png")]
    assert (out / "p02.png").exists()
    assert not (out / "p01.png").exists()


def test_render_keeps_existing_images(tmp_path):
    pdf = tmp_path / "m.pdf"
    make_pdf(pdf, [False, True])
    out = tmp_path / "out"
    render_pages(pdf, out, dpi=72)
    marker = Image.new("RGB", (7, 7), "red")
    marker.save(out / "p02.png")  # 假裝是之前轉好的圖

    assert render_pages(pdf, out, dpi=72) == [PageImage(pdf_page=2, path=out / "p02.png")]
    assert Image.open(out / "p02.png").size == (7, 7)  # 沒有被覆寫


def test_page_paths(tmp_path):
    assert page_filename(9) == "p09.png"
    assert page_image_path(tmp_path, "ac", 11) == tmp_path / "ac" / "p11.png"
